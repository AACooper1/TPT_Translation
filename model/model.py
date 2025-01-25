# Python tool imports
import sys, os, debugpy
from numpy import argmax
import pickle as pkl

# HuggingFace imports
from accelerate import Accelerator, load_checkpoint_and_dispatch
from accelerate.utils import tqdm as acc_tqdm
from accelerate.logging import get_logger
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

from transformers import AutoConfig, MarianTokenizer, AutoModel, MarianMTModel

from datasets import load_dataset

#PyTorch imports
from torch.optim import AdamW
from torch.nn.functional import cross_entropy
from torch.distributed.elastic.multiprocessing.errors import record, ErrorHandler
from torch.utils.data import DataLoader, Dataset
from torch import select, Tensor, stack, empty, load
from os.path import exists

# File imports
from tpt_rewrite import TreePlantedHead
from test import compute_metrics

CKPT_DIR = "/home/alexis/TPT_new/TPT_Translation/checkpoints"

def train_tpt():
    accelerator = Accelerator(project_dir='checkpoints', mixed_precision='fp16')
    device = accelerator.device

    model = MarianMTModel(AutoConfig.from_pretrained('Helsinki-NLP/opus-mt-mul-en'))

    model.config.num_hidden_layers = 6
    model.config.num_attention_heads = 4


    if exists(CKPT_DIR):
        try:
            accelerator.load_state(CKPT_DIR)
        except:
            print("Failed to load checkpoint at", CKPT_DIR + '.')

    lr = 1e-4
    n_epochs = 10
    batch_size = 128
    λ = 0.5
    n_heads = 1
    max_len = 128

    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-mul-en", legacy=False)
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_function = cross_entropy

    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.pad_token_id = tokenizer.eos_token_id

    loss_history = {}

    def truncate_sequence(input):
        input["la"] = input["la"][:max_len]
        input["en"] = input["en"][:max_len]
        return input

    def tokenize(input):
        return tokenizer(input["la"], text_target=input["en"], return_tensors='pt', padding='max_length', max_length=max_len, truncation=True)

    dataset = load_dataset("grosenthal/latin_english_parallel", split="train")#[:1000]  #DEBUG

    tpt_encoder_head = TreePlantedHead(device, tokenizer, λ=λ, max_len=128)


    with accelerator.main_process_first():
        dataset = dataset.remove_columns("la").with_format("torch")
        dataset = dataset.add_column("la", tpt_encoder_head.preprocessed)
        dataset = dataset.map(truncate_sequence)
        dataset = dataset.map(tokenize, batched=True, batch_size=128)
        dataset = dataset.remove_columns(["file", "en", "id"]).with_format("torch")
        dataset = dataset.add_column("id", list(range(len(dataset["la"]))))
        dataset = dataset.select(list(range(len(tpt_encoder_head.supervision.keys()))))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )

    with (
        acc_tqdm(range(n_epochs), desc="Epochs", main_process_only=True, leave=False) as epochs
    ):
            
        for epoch in epochs:
            batches = acc_tqdm(
                dataloader, 
                main_process_only=True, 
                desc="Batches", 
                leave=False
            )
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
                for i in acc_tqdm(range(len(batch['input_ids'])), desc="Word weights"):
                    word_outputs.append(
                        tpt_encoder_head.token_weights_to_word_weights(
                            batch['id'][i].item(), 
                            token_outputs['encoder_attentions'][-1][i,-1,:,:]
                        )
                    )
                word_outputs = stack(word_outputs)
                
                tp_loss = tpt_encoder_head.calculate_tree_loss(word_outputs, ids, batch_size).item()

                nwp_loss = token_outputs['loss']

                loss = nwp_loss + tpt_encoder_head.λ * tp_loss / n_heads

                batches.set_postfix({"NWP Loss": f"{nwp_loss:.3f}", "Tree Loss": f"{tp_loss:.3f}"})
                loss_history[epoch].append(loss.item())
                
                accelerator.backward(loss)
                optimizer.step()
            pass

            epochs.set_postfix({"Epoch Loss": loss.item()})
            accelerator.save_state(CKPT_DIR)

        pass

@record
def main(func):
    try:
        func()
    except Exception as e:
        raise e

if __name__ == "__main__":
    debugpy.listen(("localhost", 5670 + int(os.getenv("RANK"))))

    if len(sys.argv) < 2:
        print("Argument 'train_tph' or 'train_base' is required.")

    if "train_tph" in sys.argv:
        model, loss_history, output_history = train_tpt()

    elif True:
        print("Error: sys.argv =", sys.argv)
        raise NotImplemented

    pass