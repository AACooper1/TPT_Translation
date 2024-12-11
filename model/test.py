# Python tool imports
import sys
from numpy import argmax, where, count_nonzero, mean
import pickle as pkl
import warnings
from random import sample

# HuggingFace imports
from accelerate import Accelerator, load_checkpoint_and_dispatch, init_empty_weights
from accelerate.utils.tqdm import tqdm
from accelerate.logging import get_logger
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

from transformers import AutoConfig, MT5ForConditionalGeneration, MT5Tokenizer, AutoModel

from datasets import load_dataset

import evaluate

#PyTorch imports
from torch.optim import AdamW
from torch.nn.functional import cross_entropy
from torch.nn.modules import Module
from torch.distributed.elastic.multiprocessing.errors import record, ErrorHandler
from torch.utils.data import DataLoader, Dataset
from torch import load

def tokenize(input):
    return tokenizer(input["la"], text_target=input["en"], return_tensors='pt', padding='max_length', max_length=32, truncation=True)


def predict(model, tokenizer, input):
    inputs = tokenizer(input, return_tensors='pt')
    input_ids = inputs["input_ids"].to(accelerator.device)
    mask = inputs["attention_mask"].to(accelerator.device)

    outputs = model.generate(input_ids=input_ids, attention_mask=mask, max_new_tokens=256)

    inputs_decoded = tokenizer.decode(input_ids[0])
    outputs_decoded = tokenizer.decode(outputs[0])

    return {
         "Input": inputs_decoded,
         "Output": outputs_decoded,
    }

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    preds = preds.cpu()
    labels = labels.cpu()
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = where(labels != 0, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

if __name__ == "__main__":
    accelerator = Accelerator()
    model = MT5ForConditionalGeneration(AutoConfig.from_pretrained('google/mt5-base'))
    model = accelerator.unwrap_model(model)

    # state_dict = load('results/pytorch_model.bin')
    model = load_state_dict_from_zero_checkpoint(model, 'results')

    # config = AutoConfig.from_pretrained("results")
    # with init_empty_weights():
    #     model = AutoModel.from_config(config)

    tokenizer = MT5Tokenizer.from_pretrained('google/mt5-base', legacy=False)

    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # inputs = tokenizer("Arma virumque cano Troiae qui primus ab oris Italiam Laviniaque venit littora.", return_tensors='pt').to("cuda")

    # outputs = model.generate(inputs["input_ids"])

    # result = tokenizer.decode(outputs[0])

    dataset = load_dataset("grosenthal/latin_english_parallel", split="test")
    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.remove_columns(["id", "file", "la", "en"]).with_format("torch")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model, dataloader = accelerator.prepare(model, dataloader)
    metric = evaluate.load("sacrebleu")

    if "-o" not in sys.argv:
        results = []
        predictions = []
        model.eval()

        for batch in dataloader:
            preds = model(**batch).logits.argmax(dim=-1)
            results.append(compute_metrics((preds, batch["labels"])))
            inputs = tokenizer.batch_decode(batch["input_ids"])
            outputs = tokenizer.batch_decode(preds)
            labels = tokenizer.batch_decode(batch["labels"])
            for i in range(len(batch)):
                predictions.append({"Input": inputs[i],"Prediction": outputs[i], "Target": labels[i]})

        bleu = "Average BLEU:" + str(sum([i["bleu"] for i in results]) / len(results))
        print(bleu)

        with open("results/eval_base.log", "w") as out_file:
            if accelerator.is_main_process:
                out_file.write(bleu + "\n\n")
                for i in sample(predictions, 20):
                    out_file.write(f"Input: {i['Input']}\n")
                    out_file.write(f"Target: {i['Target']}\n")
                    out_file.write(f"Prediction: {i['Prediction']}\n")
                    out_file.write("\n")
    else:
        option = ''
        while option != '-q':
            option = input("Input a sentence, or enter '-q' to quit:\n\t")
            print(predict(model, tokenizer, option))

             
