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

from transformers import AutoConfig, MT5ForConditionalGeneration, MT5Tokenizer

from datasets import load_dataset

#PyTorch imports
from torch.optim import AdamW
from torch.nn.functional import cross_entropy
from torch.distributed.elastic.multiprocessing.errors import record, ErrorHandler
from torch.utils.data import DataLoader, Dataset

# File imports
from tpt import TreePlantedHead
from test import compute_metrics


# God shut up already
warnings.filterwarnings("ignore")

CKPT_DIR = "/home/alexis/TPT_Translation/model/checkpoints/checkpoints/checkpoint_1/converted_model"
CKPT_FILE = "/home/alexis/TPT_Translation/model/checkpoints/checkpoints/checkpoint_1/converted_model/pytorch_model.bin"
TKN_FILE = "/home/alexis/TPT_Translation/model/checkpoints/checkpoints/tokenizer/tokens.pkl"

def train_tpt():
    accelerator = Accelerator()
    device = accelerator.device

    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")

    lr = 1e-4
    n_epochs = 30
    batch_size = 48
    lmbda = 1
    n_heads = 1

    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base", legacy=False)
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_function = cross_entropy

    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.pad_token_id = tokenizer.eos_token_id

    loss_history = {}

    def tokenize(input):
        return tokenizer(input["la"], text_target=input["en"], return_tensors='pt', padding='max_length', max_length=32, truncation=True)

    dataset = load_dataset("grosenthal/latin_english_parallel", split="train")
    # Not using a decoder here, maybe for the full project
    tpt_encoder_head = TreePlantedHead(dataset, tokenizer, decoder=False)

    for i in tpt_encoder_head.superv_sent_corresp:
        tpt_encoder_head.superv_sent_corresp[i].to(device)

    for i in tqdm(tpt_encoder_head.extra_data):
        dataset = dataset.add_item(
            {
                'la': tpt_encoder_head.extra_data[i]["la"],
                'en': tpt_encoder_head.extra_data[i]["en"],
                'id': i,
                'file': None
            }
        )

    with accelerator.main_process_first():
        dataset_tokenized = dataset.map(tokenize, batched=True, batch_size=128)
        dataset_tokenized = dataset_tokenized.remove_columns(["file", "en"]).with_format("torch")

    dataloader = DataLoader(dataset_tokenized, batch_size=batch_size, shuffle=True)

    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )

    with (
        tqdm(range(n_epochs), main_process_only=True, desc="Epochs", postfix={"Loss:": "N/A"}, position=1) as epochs,  
        tqdm(dataloader, main_process_only=True, desc="Batches", postfix={"Loss:": "N/A"}, position=1) as batches
        
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

                outputs = model(input_ids=inputs, labels=targets, attention_mask=mask, output_attentions=True)
                
                preds = outputs.logits.argmax(dim=-1)
                attns = outputs.encoder_attentions[-1][:,0,:,:]
                nwp_loss = outputs.loss
                
                batches.set_postfix_str({"Loss": nwp_loss.item()})

                for a in range(len(batch)):
                    if ids[a].item() in tpt_encoder_head.word_tokens:
                        # Try block because it's just too buggy tbqh
                        try:
                            tree_loss += tpt_encoder_head(attns[a], ids[a])  
                        except:
                            print("Failed to calculate tree planting loss. Adding 0.")
                            failure_count+=1
                    else:
                        pass
                if tree_loss != 0:
                    loss = tpt_encoder_head.calculate_batch_loss(
                        nwp_loss,
                        tree_loss,
                        lmbda,
                        n_heads
                    )
                else:
                    loss = nwp_loss

                loss_history[epoch].append(loss)
                
                accelerator.backward(loss)
                optimizer.step()
            
            print(f"Failures at epoch {epoch}: {failure_count}")
            if failure_count > 0.4 * len(dataset_tokenized):
                print("But the biggest failure of all here... is Alexis A. Cooper.")

            with open("logs/logs.log", "a") as out_file:
                example_output = [tokenizer.decode(i) for i in outputs.logits.argmax(dim=-1)]
                if accelerator.is_local_main_process:
                    out_file.write(f"""
                        Epoch: {epoch}; Loss: {
                                                sum(loss_history[epoch]) / 
                                                len(loss_history[epoch])
                                            }
                        Input: {[tokenizer.decode(i) for i in inputs][0]}.
                        Target: {[tokenizer.decode(i) for i in targets][0]}
                        Output: {example_output[0]}
                            """
                    )

            model.save_checkpoint("results_tpt")

            
            pass

            # epochs.set_postfix({"Loss:": sum(loss_history[epoch]) / len(loss_history[epoch])})

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
    try:
        model = MT5ForConditionalGeneration(AutoConfig.from_pretrained('google/mt5-base'))
        model = accelerator.unwrap_model(model)
        model = load_state_dict_from_zero_checkpoint(model, 'results-mc')
    except:
        model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")

    # Set hyperparameters
    lr = 1e-4
    n_epochs = 20 # This is just because I'm continuing the training from 10 - I think 30 is better than 10 in this case
    batch_size = 72

    # Load tokenizer, optimizer, and loss function
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base", legacy=False)
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
        print("Argument 'train' or 'test' is required.")

    if "train_tph" in sys.argv:
        model, loss_history, output_history = train_tpt()

    if "train_base" in sys.argv:
        model, loss_history, output_history = train_base(truncated_data=False)
        pass

    elif True:
        print("Error: sys.argv =", sys.argv)
        raise NotImplemented

    pass