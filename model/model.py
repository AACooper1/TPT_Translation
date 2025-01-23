# Python tool imports
import sys
from numpy import argmax
import pickle as pkl
import warnings

# HuggingFace imports
from accelerate import Accelerator, load_checkpoint_and_dispatch
from accelerate.utils.tqdm import tqdm
from accelerate.logging import get_logger
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

from transformers import AutoConfig, MarianTokenizer, AutoModel, MarianMTModel

from datasets import load_dataset

#PyTorch imports
from torch.optim import AdamW
from torch.nn.functional import cross_entropy
from torch.distributed.elastic.multiprocessing.errors import record, ErrorHandler
from torch.utils.data import DataLoader, Dataset
from torch import select, Tensor, stack, empty

# File imports
from tpt_rewrite import TreePlantedHead
from test import compute_metrics

CKPT_DIR = "/home/alexis/TPT_Translation/model/checkpoints/checkpoints/checkpoint_1/converted_model"
CKPT_FILE = "/home/alexis/TPT_Translation/model/checkpoints/checkpoints/checkpoint_1/converted_model/pytorch_model.bin"
TKN_FILE = "/home/alexis/TPT_Translation/model/checkpoints/checkpoints/tokenizer/tokens.pkl"

def train_tpt():
    accelerator = Accelerator()
    device = accelerator.device

    model = MarianMTModel(AutoConfig.from_pretrained('Helsinki-NLP/opus-mt-mul-en'))

    lr = 1e-2
    n_epochs = 30
    batch_size = 48
    lmbda = 1
    n_heads = 1

    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-mul-en", legacy=False)
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_function = cross_entropy

    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.pad_token_id = tokenizer.eos_token_id

    loss_history = {}

    def tokenize(input):
        return tokenizer(input["la"], text_target=input["en"], return_tensors='pt', padding='max_length', max_length=64, truncation=True)

    dataset = load_dataset("grosenthal/latin_english_parallel", split="train")#[:1000]  #DEBUG

    tpt_encoder_head = TreePlantedHead(tokenizer)

    with accelerator.main_process_first():
        # dataset = dataset.remove_columns("la").with_format("torch")
        dataset = dataset.add_column("la_", tpt_encoder_head.preprocessed)
        dataset = dataset.map(tokenize, batched=True, batch_size=128)
        dataset = dataset.remove_columns(["file", "en", "id"]).with_format("torch")
        dataset = dataset.add_column("id", list(range(len(dataset["la"]))))
        dataset = dataset.select(list(range(len(tpt_encoder_head.supervision.keys()))))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )

    with (
        tqdm(range(n_epochs), desc="Epochs", position=0) as epochs,  
        tqdm(dataloader, main_process_only=True, desc="Batches", position=1) as batches        
    ):
            
        for epoch in range(n_epochs):
            loss_history[epoch] = []
            tree_loss = 0
            failure_count = 0
            for batch in batches:                
                optimizer.zero_grad()

                inputs = batch["input_ids"]
                targets = batch["labels"]
                ids = batch["id"]
                mask = batch["attention_mask"]

                token_outputs = model(input_ids=inputs, labels=targets, attention_mask=mask, output_attentions=True)

                # Get word-level weights
                word_outputs = []
                for i in range(len(batch['input_ids'])):
                    word_outputs.append(
                        tpt_encoder_head.token_weights_to_word_weights(
                            batch['id'][i].item(), 
                            token_outputs['encoder_attentions'][-1][i,-1,:,:]
                        )
                    )
                word_outputs = stack(word_outputs)
                
                tp_loss = tpt_encoder_head.calculate_tree_loss(word_outputs, ids).item()

                nwp_loss = token_outputs['loss']

                loss = nwp_loss + tpt_encoder_head.λ * tp_loss

                batches.set_postfix({"Loss": f"{loss:.3f}"})
                loss_history[epoch].append(loss)
                
                accelerator.backward(loss)
                optimizer.step()
            pass

            # epochs.set_postfix({"Loss:": sum(loss_history[epoch]) / len(loss_history[epoch])})
        pkl.dump(model)

        pass

def train_base(truncated_data=False):
    # Set up logging
    # logger = get_logger(__name__, log_level="ERROR")
    # logger.setLevel("ERROR")
    loss_history = {}
    output_history = {}

    # Prepare accelerator
    accelerator = Accelerator()
    device = accelerator.device

    # Load model - from checkpoint or pretrained (turns out 100k isn't enough wow no way????????)
    # try:
    #     # model = MT5ForConditionalGeneration(AutoConfig.from_pretrained('google/mt5-small'))
    #     model = accelerator.unwrap_model(model)
    #     model = load_state_dict_from_zero_checkpoint(model, 'results-mc')
    # except:
    model = AutoModel.from_config(AutoConfig.from_pretrained('Helsinki-NLP/opus-mt-mul-en'))

    # Set hyperparameters
    lr = 1e-4
    n_epochs = 20 # This is just because I'm continuing the training from 10 - I think 30 is better than 10 in this case
    batch_size = 72

    # Load tokenizer, optimizer, and loss function
    tokenizer = MarianTokenizer.from_pretrained("google/mt5-small", legacy=False)
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_function = cross_entropy
    # open("logs/logs.log", "w")

    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.pad_token_id = tokenizer.eos_token_id

    def tokenize(input):
        return tokenizer(input["la"], text_target=input["en"], return_tensors='pt', padding='max_length', max_length=32, truncation=True)

    # Load dataset, tokenize, and batch in dataloader (use truncated if debugging)
    split = "train"
    dataset = load_dataset("grosenthal/latin_english_parallel", split=split)
    dataset = dataset.select(range(100)) if truncated_data else dataset
    with accelerator.main_process_first():
        try:
            with open(TKN_FILE, "rb+") as tokens:
                pk = pkl.Unpickler(tokens)
                dataset = pkl.loads(tokens)
                print("Loaded tokens from file!")
        except:
            if not accelerator.is_local_main_process:
                print("tokens.pkl not found. Tokenizing...")
            dataset = dataset.map(tokenize, batched=True)
            with open(TKN_FILE, "wb+") as out_file:
                with accelerator.main_process_first():
                    pk = pkl.Pickler(out_file)
                    pk.dump(dataset)
    

    dataset = dataset.remove_columns(["id", "file", "la", "en"]).with_format("torch")
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Move all to accelerator
    model.to(device)
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    with (
        tqdm(range(n_epochs), main_process_only=True, desc="Epochs", postfix={"Loss:": "N/A"}) as epochs,  
        tqdm(train_dataloader, main_process_only=True, desc="Batches", postfix={"Loss:": "N/A"}) as batches
        
    ):
        epochs.set_postfix_str("N/A")
        batches.set_postfix_str("N/A")

        for epoch in epochs:
            loss_history[epoch] = []
            
            for batch in batches:                
                optimizer.zero_grad()
                inputs = batch["input_ids"]
                targets = batch["labels"]
                mask = batch["attention_mask"]
                outputs = model(input_ids=inputs, attention_mask=mask, labels=targets)
                loss = outputs.loss
                batches.set_postfix_str({"Loss": loss.item()})
                accelerator.backward(loss)
                optimizer.step()
                
                # update the progress bars and loss history
                # batches.set_postfix({"Loss:": loss.item()})
                loss_history[epoch].append(loss.item())

            with open("logs/logs.log", "a") as out_file:
                example_output = [tokenizer.decode(i) for i in outputs.logits.argmax(dim=-1)]
                if accelerator.is_local_main_process:
                    out_file.write(f"""
                        Epoch: {epoch}; Loss: {sum(loss_history[epoch]) / len(loss_history[epoch])}
                        Input: {[tokenizer.decode(i) for i in inputs][0]}.
                        Target: {[tokenizer.decode(i) for i in targets][0]}
                        Output: {example_output[0]}
                                """)

                    
                    # output_history[epoch] = {
                    #     "output": tokenizer.batch_decode(model.generate(inputs[0]))
                    # }
            model.save_checkpoint("results")

            
            pass

            epochs.set_postfix({"Loss:": sum(loss_history[epoch]) / len(loss_history[epoch])})

    with open("results/log_history_tp.pk", "wb") as out_file:
        pk = pkl.Pickler
        pk.dump(loss_history)
    return model, loss_history, output_history

@record
def main(func):
    try:
        func()
    except Exception as e:
        raise e

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Argument 'train_tph' or 'train_base' is required.")

    if "train_tph" in sys.argv:
        model, loss_history, output_history = train_tpt()

    if "train_base" in sys.argv:
        model, loss_history, output_history = train_base(truncated_data=False)
        pass

    elif True:
        print("Error: sys.argv =", sys.argv)
        raise NotImplemented

    pass