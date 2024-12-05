# Python imports
import argparse
import random as rd
import re
from tqdm import tqdm

# Program imports
from tpt import TreePlantedHead
import pyconll 

# Model imports
import torch 
from transformers import MT5Tokenizer, MT5ForConditionalGeneration, AutoConfig, Seq2SeqTrainer, Seq2SeqTrainingArguments, logging 
from torch.utils.data import DataLoader 
from datasets import load_dataset 
from accelerate import Accelerator

# Global constants
CKPT_PATH = "model/checkpoints"

def train_base_model(model: MT5ForConditionalGeneration, tokenizer: MT5Tokenizer, accelerator: Accelerator):
    # Set up tokenizer function
    def tokenize(input):
        tokenized = tokenizer(
            input["input_ids"], 
            text_target=input["labels"], 
            return_tensors="pt", 
            max_length=128,
            padding="max_length", 
            truncation=True
        )
        
        return tokenized

    # Set up model and dataset
    
    logging.set_verbosity_info()

    default_args = {
    "eval_strategy": "steps",
    "num_train_epochs": 1,
    "log_level": "info",
    "report_to": "none",
    "save_strategy": "epoch",
    "output_dir": "model/checkpoints",
    "overwrite_output_dir":True,
}

    dataset = load_dataset("grosenthal/latin_english_parallel").rename_columns({"la": "input_ids", "en": "labels"})
    dataset.set_format(type="torch", columns=["input_ids", "labels"])

    training_args = Seq2SeqTrainingArguments(per_device_train_batch_size=8, **default_args)

    # Model hyperparameters (same as original TPT paper)
    learn_rate = 5e-5
    n_epochs = 10
    dropout_rate = 0.1
    batch_size=8

    train_dataloader = DataLoader(
        dataset["train"].map(tokenize, batch_size=10, batched=True), 
        shuffle=True, 
        batch_size=batch_size)
    
    test_dataloader = DataLoader(
        dataset["test"].map(tokenize, batch_size=10, batched=True), 
        shuffle=True, 
        batch_size=batch_size)
    optimizer = torch.optim.AdamW(model.parameters, lr=learn_rate)

    # HF Accelerator setup
    model, optimizer, training_dataloader = accelerator.prepare(
        model, optimizer, training_dataloader
    )

    # Main training loop
    total_progress_bar = tqdm(range(len(train_dataloader) * n_epochs), position=0)

    for epoch in tqdm(range(n_epochs), position=1, desc="Epoch"): 
        for batch in tqdm(train_dataloader, position=2, leave=False, desc="Batch"):
            optimizer.zero_grad()
            inputs, targets = batch
            outputs = model(inputs)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()

            accelerator.save_state(CKPT_PATH)

            total_progress_bar.update(1)


    return model

#################################################
# ==== === TREE-PLANTED HEAD DEFINITION === === #
#################################################

# Use GPT-2 Small BPE and architecture, as in paper -- modified to be encoder-decoder? Or just use MT5 or similar encoder-decoder model
def main():
    treebank_filepath = 'UD/UD_Latin-Perseus/la_perseus-ud-train.conllu'
    treebank = pyconll.load_from_file(treebank_filepath)
    treebank_filepath = 'UD/UD_Latin-CIRCSE/la_circse-ud-test.conllu'
    treebank += pyconll.load_from_file(treebank_filepath)

    parser = argparse.ArgumentParser(
        prog='Tree-Planted Translation',
        description="A machine translation model incorporating Yoshida et al.'s (2024) 'Tree-Planted Transformer structure."
    )

    # tokenizer = MT5Tokenizer.from_pretrained("google/mt5-large")
    # model = MT5ForConditionalGeneration.from_pretrained("google/mt5-large")

    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")
    cfg = AutoConfig.from_pretrained("google/mt5-base")
    model = MT5ForConditionalGeneration(cfg)
    # model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
    model.config.output_attentions = True

    tph = TreePlantedHead()

    # prefix = "translate English to Latin: "
    # context = "en: When Red wins, she stands alone. la: Cum vincit Rubra, sola stat."
    
    # prompt = prefix + context + "When Blue wins--which is always--she moves on to the next thing. la:"

    # target = "Cum vincit Caerulea--qui fit semper--ad proximum procedit."

    tree = treebank[1590] #rd.choice(treebank)

    distance_matrix = tph.pyconll_to_distance_matrix(tree, debug=False)
    supervision_matrix = tph.distance_matrix_to_supervision(distance_matrix, decoder=False)    

    prompt = ' '.join([re.sub(r'[0-9]+', '', word) for word in list(distance_matrix.keys())])


    input = tokenizer([prompt], return_tensors="pt")
    input_ids = input.input_ids
    input_word_level = tph.subword_compat(prompt, tokenizer)
    decoder_ids = input.attention_mask
    output = model(input_ids, return_dict=True, decoder_input_ids=decoder_ids, output_attentions=True)
    decoder_attentions = output.decoder_attentions
    encoder_attentions = output.encoder_attentions

    subword_to_word_test = tph.subtoken_weights_to_word_weights(input_word_level, encoder_attentions[0][0])
    input_words = ""
    for word in input_word_level:
        for token in word:
            input_words += tokenizer.decode(token)
        input_words += " "

    output_decoded = tokenizer.decode(output.logits.argmax(dim=-1)[0])
    input_decoded = [tokenizer.decode(input_ids[0][i]) for i in range(len(input_ids[0]))]
    print(f"WORD {input_decoded[1]}:")
    the = dict(zip(input_decoded, subword_to_word_test[1]))
    for i in the:
        print(f"{i} : {the[i]}")

    TPLoss = tph.calculate_tree_loss(supervision_matrix, subword_to_word_test)

    try:
        print(TPLoss.item())
    except:
        print(TPLoss)

    pass

from pynvml import * 

# Copied from HF
def gpu_utilization():
    nvmlInit() 
    handle = nvmlDeviceGetHandleByIndex(0) 
    info = nvmlDeviceGetMemoryInfo(handle) 
    return f"GPU memory occupied: {info.used//1024**2} MB."


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print(gpu_utilization()) 


if __name__ == '__main__':
    accelerator = Accelerator()

    try:
        model = MT5ForConditionalGeneration.from_pretrained(CKPT_PATH)
        accelerator.load_state(CKPT_PATH)
    except:
        print(f"No checkpoints found in {CKPT_PATH}. Creating new model from scratch...")
        model = None

    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")
    cfg = AutoConfig.from_pretrained("google/mt5-base")
    model = MT5ForConditionalGeneration(cfg).to(accelerator.device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    print(gpu_utilization())
    
    train_base_model(model, tokenizer, accelerator)