# Python tool imports
import sys
from numpy import argmax
import pickle as pkl
import warnings

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
from torch import select, Tensor, stack, empty

# File imports
from tpt_rewrite import TreePlantedHead
from test import compute_metrics

CKPT_DIR = "/home/alexis/TPT_Translation/model/checkpoints/checkpoints/checkpoint_1/converted_model"
CKPT_FILE = "/home/alexis/TPT_Translation/model/checkpoints/checkpoints/checkpoint_1/converted_model/pytorch_model.bin"
TKN_FILE = "/home/alexis/TPT_Translation/model/checkpoints/checkpoints/tokenizer/tokens.pkl"

def train_tpt():
    accelerator = Accelerator(project_dir='checkpoints')
    device = accelerator.device

    model = MarianMTModel(AutoConfig.from_pretrained('Helsinki-NLP/opus-mt-mul-en'))

    lr = 1e-4
    n_epochs = 30
    batch_size = 128
    λ = 0.5
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

    tpt_encoder_head = TreePlantedHead(device, tokenizer, λ=λ)


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
                for i in range(len(batch['input_ids'])):
                    word_outputs.append(
                        tpt_encoder_head.token_weights_to_word_weights(
                            batch['id'][i].item(), 
                            token_outputs['encoder_attentions'][-1][i,-1,:,:]
                        )
                    )
                word_outputs = stack(word_outputs)
                
                tp_loss = tpt_encoder_head.calculate_tree_loss(word_outputs, ids, batch_size).item()

                nwp_loss = token_outputs['loss']

                loss = nwp_loss + tpt_encoder_head.λ * tp_loss

                batches.set_postfix({"Loss": f"{loss:.3f}"})
                loss_history[epoch].append(loss.item())
                
                accelerator.backward(loss)
                optimizer.step()
            pass

            epochs.set_postfix({"Loss:": loss.item()})
            model.save_state()

        pass

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

    elif True:
        print("Error: sys.argv =", sys.argv)
        raise NotImplemented

    pass