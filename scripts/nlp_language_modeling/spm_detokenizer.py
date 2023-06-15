import argparse
import gzip
import json
import multiprocessing
import os
import pathlib
import sys
import time

import ftfy
import numpy as np
import torch

from nemo.collections.nlp.data.language_modeling.megatron import indexed_dataset
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

try:
    import nltk

    nltk_available = True
except ImportError:
    nltk_available = False

def get_tokenizer(args):
    tokenizer = get_nmt_tokenizer(
        library=args.tokenizer_library,
        tokenizer_model=args.tokenizer_model,
    )
    return tokenizer

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to the input npy`',
    )
    group.add_argument(
        '--output',
        type=str,
        default="output.txt",
        help='Path to the input npy`',
    )
    group = parser.add_argument_group(title='tokenizer')
    group.add_argument(
        '--tokenizer-library',
        type=str,
        required=True,
        choices=['yttm', 'sentencepiece', 'megatron', 'huggingface', 'tabular'],
        help='What tokenizer library to use.',
    )
    group.add_argument(
        '--tokenizer-model', type=str, default=None, help='Path to tokenizer model.',
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    output_tokens = np.load(args.input)
    print(f"output token shape: {output_tokens.shape}")

    tokenizer = get_tokenizer(args)
    outputs = []
    for token in output_tokens:
        out = tokenizer.ids_to_text(token)
        outputs.append(out)

    with open(args.output, 'w') as fh:
        fh.writelines(outputs)
