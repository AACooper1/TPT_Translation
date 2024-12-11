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

# God shut up already
warnings.filterwarnings("ignore")

CKPT_DIR = "/home/alexis/TPT_Translation/model/checkpoints/checkpoints/checkpoint_1/converted_model"
CKPT_FILE = "/home/alexis/TPT_Translation/model/checkpoints/checkpoints/checkpoint_1/converted_model/pytorch_model.bin"
TKN_FILE = "/home/alexis/TPT_Translation/model/checkpoints/checkpoints/tokenizer/tokens.pkl"

def train_base_model(model, tokenizer, accelerator):
    if True:
        return
    def tokenize(batch):
        return tokenizer(
            batch["input_ids"],
            text_target=batch["labels"],
            max_length=128,
            padding="max_length",
            truncation=True
        )

    dataset = load_dataset("grosenthal/latin_english_parallel")
    dataset = dataset.rename_columns({"la": "input_ids", "en": "labels"})

    train_dataset = dataset["train"].map(tokenize, batched=True, remove_columns=dataset["train"].column_names)
    test_dataset = dataset["test"].map(tokenize, batched=True, remove_columns=dataset["test"].column_names)

    learn_rate = 5e-5
    n_epochs = 10
    batch_size = 16

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        model=model, 
        padding="longest", 
        return_tensors="pt"
    )

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, collate_fn=data_collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learn_rate)
    accelerator.clip_grad_norm = 1.0

    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader
    )

    os.makedirs(CKPT_DIR, exist_ok=True)

    total_progress_bar = tqdm(range(len(train_dataloader) * n_epochs), position=0)

    for epoch in tqdm(range(n_epochs), position=1, desc="Epoch"):
        model.train()
        for batch in tqdm(train_dataloader, position=2, leave=False, desc="Batch"):
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits
            accelerator.backward(loss)
            optimizer.step()
            total_progress_bar.update(1)

        if (epoch + 1) % 5 == 0:
            accelerator.save_state(CKPT_DIR)
            print(f"Checkpoint saved at epoch {epoch + 1}")

    return model

def train(truncated_data=False):
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
        model = MT5ForConditionalGeneration(AutoConfig.from_pretrained('grosenthal/mbart_la_en'))
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
    open("logs/logs.log", "w")

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
                # if accelerator.is_local_main_process:
                #     print(f"Epoch {epoch} finished. Writing logs...")
                # example_output = tokenizer.decode(model.generate(**tokenizer("Salutem dico.", return_tensors="pt").to(device), max_new_tokens=50)[0])

                # if accelerator.is_local_main_process:
                #     out_file.write(f"""
                #         Epoch: {epoch}; Loss: {sum(loss_history[epoch]) / len(loss_history[epoch])}
                #         Input: Salutem dico.
                #         Output: {example_output}
                #                 """)
                # example_output = tokenizer.decode(model.generate(**tokenizer("Flagitiis et fraude cano damnabile carmen, de rationibus ex ore stultis bene docti.", return_tensors="pt").to(device), max_new_tokens=50)[0])
                # if accelerator.is_local_main_process:
                #     out_file.write(f"""
                #         Epoch: {epoch}; Loss: {sum(loss_history[epoch]) / len(loss_history[epoch])}
                #         Input: Flagitiis et fraude cano damnabile carmen, de rationibus ex ore stultis bene docti.
                #         Output: {example_output}
                #                 """)
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

    with open("results/log_history.pk", "wb") as out_file:
        pk = pkl.Pickler
        pk.dump(loss_history)
    return model, loss_history, output_history

@record
def main(func):
    error_handler = ErrorHandler
    try:
        func()
    except Exception as e:
        raise e

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Argument 'train' or 'test' is required.")

    if sys.argv[1] == "train":
        model, loss_history, output_history = train(truncated_data=False)
        pass
    elif True:
        print("Error: sys.argv =", sys.argv)
        raise NotImplemented

    elif False:
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch
        from accelerate.inference import prepare_pippy

        with init_empty_weights():
            model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")

        model = load_checkpoint_and_dispatch(
            model, checkpoint=CKPT_FILE, device_map="auto"
        )

        tokenizer: MT5Tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")

        input = tokenizer.encode("Salutem dico.", return_tensors='pt')
        
        example_inputs = {"input_ids": input}
        model = prepare_pippy(model, example_args=(input,))
    
    elif False:
        model = MT5ForConditionalGeneration.from_pretrained("/home/alexis/TPT_Translation/model/checkpoints/checkpoints/checkpoint_0/converted_model", ignore_mismatched_sizes=True) 
        print("Loaded model!")

        tokenizer: MT5Tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")
        input = tokenizer.encode("Salutem dico.", return_tensors='pt')
        print("Tokenized!")

        outputs = model.generate(input, decoder_input_ids=input, max_new_tokens=20)
        outputs = outputs[0]
        print("Finished inference!")

        result = tokenizer.decode(outputs)
        print(result)

    pass