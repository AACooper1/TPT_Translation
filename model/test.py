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

# File imports
from tpt import TreePlantedHead

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

    dataset = load_dataset("grosenthal/latin_english_parallel", split="train")
    tpt_encoder_head = TreePlantedHead(dataset, tokenizer, decoder=False)
    tpt_decoder_head = TreePlantedHead(dataset, tokenizer, decoder=True)

    # with open(f"model/tpt.pkl", 'wb+') as in_file:
    #     unpk = pkl.Pickler(in_file)
    #     unpk.dump(tpt_encoder_head)

    for i in tqdm(tpt_encoder_head.extra_data):
        dataset = dataset.add_item(
            {
                'la': tpt_encoder_head.extra_data[i]["la"],
                'en': tpt_encoder_head.extra_data[i]["en"],
                'id': i,
                'file': None
            }
        )
    dataset_tokenized = dataset.map(tokenize, batched=True)
    dataset_tokenized = dataset_tokenized.remove_columns(["file", "en"]).with_format("torch")

    # with open(f"model/dataset.pkl", 'rb') as out_file:
    #         pk = pkl.Unpickler(out_file)
    #         dataset_tokenized = pk.load()

    dataloader = DataLoader(dataset_tokenized, batch_size=32, shuffle=True)

    model, dataloader = accelerator.prepare(model, dataloader)
    metric = evaluate.load("sacrebleu")

    if "-o" not in sys.argv:
        results = []
        predictions = []
        model.eval()

        for batch in dataloader:
            out = model(input_ids=batch["input_ids"], labels=batch["labels"], attention_mask=batch["attention_mask"], output_attentions=True)
            preds = out.logits.argmax(dim=-1)
            attns = out.encoder_attentions[-1][:,0,:,:]
            results.append(compute_metrics((preds, batch["labels"])))

            trzy = [tpt_encoder_head.superv_sent_corresp[i] if i in tpt_encoder_head.superv_sent_corresp else None for i in batch["la"]]

            inputs = tokenizer.batch_decode(batch["input_ids"])
            outputs = tokenizer.batch_decode(preds)
            labels = tokenizer.batch_decode(batch["labels"])
            
            tree_loss = 0
            for a in range(len(batch)):
                tree_loss += tpt_encoder_head(attns[a]) if attns[a] is not None else 0
                pass

            for i in range(len(batch)):
                predictions.append({"Input": inputs[i],"Prediction": outputs[i], "Target": labels[i]})

        bleu = "Average BLEU:" + str(sum([i["bleu"] for i in results]) / len(results))
        print(bleu)

        with open("results/eval_tp.log", "w") as out_file:
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

             
