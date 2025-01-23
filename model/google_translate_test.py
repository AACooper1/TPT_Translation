from googletrans import Translator
from datasets import load_dataset
import evaluate
from tqdm import tqdm
from random import sample

metric = evaluate.load("sacrebleu")

dataset = load_dataset("grosenthal/latin_english_parallel", split="test")

translator = Translator()

results = []
targets = []
predictions = []

bleu = evaluate.load("sacrebleu")
meteor = evaluate.load("meteor")
gleu = evaluate.load("google_bleu")

for i in tqdm(range(len(dataset))):
    output = translator.translate(dataset[i]["la"], src='la', dest='en').text
    target = dataset[i]["en"]

    predictions.append(output)
    targets.append(target) 
    results.append({"Input": dataset[i]["la"],"Prediction": output, "Target": target})

result_b = bleu.compute(predictions=predictions, references=targets)
result_m = meteor.compute(predictions=predictions, references=targets)
result_g  = gleu.compute(predictions=predictions, references=targets)

# results_str = "Average BLEU: " + str(sum([i["bleu"] for i in results]) / len(results)) + "\nAverage METEOR: " + str(sum([i["meteor"] for i in results]) / len(results)) + "\nAverage GLEU: " + str(sum([i["gleu"] for i in results]) / len(results))

with open("final_results/eval_google.log", "w+") as out_file:
    out_file.write(f"Average BLEU: {result_b}\nAverage METEOR: {result_m}\nAverage GLEU: {result_g}")
    for i in sample(results, 20):
        out_file.write(f"Input: {i['Input']}\n")
        out_file.write(f"Target: {i['Target']}\n")
        out_file.write(f"Prediction: {i['Prediction']}\n")
        out_file.write("\n")

pass