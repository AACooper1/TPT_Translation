# Python imports
import argparse

# Program imports
import tpt

# Model imports
import torch
from transformers import MT5Tokenizer, MT5ForConditionalGeneration, AutoConfig
# from transformers import MT5ForConditionalGeneration, MT5Tokenizer

#################################################
# ==== === TREE-PLANTED HEAD DEFINITION === === #
#################################################

# Use GPT-2 Small BPE and architecture, as in paper -- modified to be encoder-decoder? Or just use MT5 or similar encoder-decoder model
def main():
    parser = argparse.ArgumentParser(
        prog='Tree-Planted Translation',
        description="A machine translation model incorporating Yoshida et al.'s (2024) 'Tree-Planted Transformer structure."
    )

    # tokenizer = MT5Tokenizer.from_pretrained("google/mt5-large")
    # model = MT5ForConditionalGeneration.from_pretrained("google/mt5-large")

    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")
    cfg = AutoConfig.from_pretrained("google/mt5-base")
    model = MT5ForConditionalGeneration(cfg)

    prefix = "Translate English to Latin.\n\n"
    context = "English:\tWhen Red wins, she stands alone.\n\n Latin:\t\tCum vincit Rubra, sola stat.\n\nEnglish:\t"
    
    prompt = prefix + context+ "When Blue wins--which is always--she moves on to the next thing.\n\nLatin:\t\t"

    input = tokenizer([prompt], return_tensors="pt")
    output = model.generate(**input, max_new_tokens=15, temperature=0.01)

    output_decoded = tokenizer.decode(output[0])
    print(prompt, end="")
    print(output_decoded)

    pass
    

if __name__ == '__main__':
    main()