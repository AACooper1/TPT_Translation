# Python imports
import argparse

# Model imports
import torch
from transformers import AutoConfig

# Use GPT-2 Small BPE and architecture.
def main():
    parser = argparse.ArgumentParser(
        prog='Tree-Planted Translation',
        description="A machine translation model incorporating Yoshida et al.'s (2024) 'Tree-Planted Transformer structure."
    )

    tokenizer_name = 'GPT2Tokenizer'
    model_name = 'GPT2LMHeadModel'

    my_config = AutoConfig.from_pretrained("model_name", n_heads=12)
    
    

if __name__ == 'main':
    main()