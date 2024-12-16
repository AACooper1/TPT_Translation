# Python tool imports
import sys
from numpy import where, count_nonzero, mean
from random import sample
from tqdm import tqdm

from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset

import evaluate


def tokenize(input):
    return tokenizer(input["la"], text_target=input["en"], return_tensors='pt', padding='max_length', max_length=32, truncation=True)


def predict(model, tokenizer, input):
    inputs = tokenizer(input, return_tensors='pt')
    input_ids = inputs["input_ids"]
    mask = inputs["attention_mask"]

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

    result_b = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    result_m = meteor.compute(predictions=decoded_preds, references=decoded_labels)
    result_g  = gleu.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result_b["score"], "meteor": result_m["meteor"], "gleu": result_g["google_bleu"]}

    prediction_lens = [count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

if __name__ == "__main__":
    model = MT5ForConditionalGeneration.from_pretrained('global_step5175')

    tokenizer = MT5Tokenizer.from_pretrained('google/mt5-base', legacy=False)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = load_dataset("grosenthal/latin_english_parallel", split="test")
    # tpt_encoder_head = TreePlantedHead(dataset, tokenizer, decoder=False)
    # tpt_decoder_head = TreePlantedHead(dataset, tokenizer, decoder=True)

    dataset_tokenized = dataset.map(tokenize, batched=True)
    dataset_tokenized = dataset_tokenized.remove_columns(["file", "en"]).with_format("torch")

    dataloader = DataLoader(dataset_tokenized, batch_size=8, shuffle=True)

    bleu = evaluate.load("sacrebleu")
    meteor = evaluate.load("meteor")
    gleu = evaluate.load("google_bleu")

    if "-o" not in sys.argv:
        results = []
        predictions = []
        model.eval()

        for batch in tqdm(dataloader):
            out = model(input_ids=batch["input_ids"], labels=batch["labels"], attention_mask=batch["attention_mask"], output_attentions=True)
            preds = out.logits.argmax(dim=-1)
            attns = out.encoder_attentions[-1][:,0,:,:]
            results.append(compute_metrics((preds, batch["labels"])))

            inputs = tokenizer.batch_decode(batch["input_ids"])
            outputs = tokenizer.batch_decode(preds)
            labels = tokenizer.batch_decode(batch["labels"])
            

            for i in range(len(batch)):
                predictions.append({"Input": inputs[i],"Prediction": outputs[i], "Target": labels[i]})

        results_str = "Average BLEU: " + str(sum([i["bleu"] for i in results]) / len(results)) + "\nAverage METEOR: " + str(sum([i["meteor"] for i in results]) / len(results)) + "\nAverage GLEU: " + str(sum([i["gleu"] for i in results]) / len(results))
        print(results_str)

        with open("final_results/eval_base30.log", "w+") as out_file:
            out_file.write(results_str + "\n\n")
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

             
