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
    n_epochs = 20 # This is just because I'm continuing the training from 10 - I think 30 is better than 10 in this case

    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base", legacy=False)
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_function = cross_entropy

    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.pad_token_id = tokenizer.eos_token_id

    with open("model/dataset.pkl", "rb") as in_file:
        pk = pkl.Unpickler(in_file)
        dataset = pk.load()
        print("Loaded dataset from file!")

    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    with open("model/tph.pkl", "rb") as in_file:
        pk = pkl.Unpickler(in_file)
        tph = pk.load()
        print("Loaded tph encoder from file!")

    for i in tph.superv_sent_corresp:
        tph.superv_sent_corresp[i].to(device)

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    with (
        tqdm(range(n_epochs), main_process_only=True, desc="Epochs", postfix={"Loss:": "N/A"}) as epochs,  
        tqdm(train_dataloader, main_process_only=True, desc="Batches", postfix={"Loss:": "N/A"}) as batches
        
    ):
            
        for epoch in range(n_epochs):
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
            model.save_checkpoint("results_tpt")

            
            pass

            epochs.set_postfix({"Loss:": sum(loss_history[epoch]) / len(loss_history[epoch])})

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