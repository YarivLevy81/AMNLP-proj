import argparse
import os
from glob import glob
from tqdm import tqdm

import pandas as pd
import tensorflow.compat.v1 as tf
import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    args = parser.parse_args()
    return args


def main(args):
    for subdir, dirs, files in os.walk(args.input_path):
        for filename in files:
            filepath = subdir + os.sep + filename
            if filepath.endswith("pt"):
                data = torch.load(filepath)
                tf.print(f'{filepath}: found {len(data)} examples')
    tf.print('done')

if __name__ == '__main__':
    args = get_args()
    tf.print(args)
    main(args)