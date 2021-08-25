import argparse
import json
import os
from glob import glob
from tqdm import tqdm
import ntpath

import pandas as pd
import numpy as np
import torch

import sys
sys.path.append('../tokenization')
import tokenization


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default='t5-small', required=False)
    parser.add_argument("--cache_dir", type=str, default=None, required=False)
    parser.add_argument("--tensor_length", type=int, default=512, required=False)
    parser.add_argument("--max_number_of_examples", type=int, default=-1, required=False)
    parser.add_argument("--label_ignore_id", type=int, default=-100, required=False)
    args = parser.parse_args()
    return args


def read_mrqa(path):
    data = []
    with open(path) as f:
        header = f.readline()
        for line in f:
            data_point = json.loads(line)
            data.append(data_point)
    return data, header


def create_example(input_ids, label_ids, tensor_length=512, label_ignore_id=-100):
    input_len = len(input_ids)
    input_mask = [1 if input_ids[i] != 0 else 0 for i in range(len(input_ids))]
    label_ids = [label_ignore_id] + label_ids

    if input_len > tensor_length or input_len != len(input_mask) or \
            len(label_ids) > tensor_length:
        print("skipping...")
        return None

    input_ids += [0] * (tensor_length - len(input_ids))
    input_mask += [0] * (tensor_length - len(input_mask))
    label_ids += [label_ignore_id] * (tensor_length - len(label_ids))

    assert len(input_ids) == tensor_length
    assert len(input_mask) == tensor_length
    assert len(label_ids) == tensor_length

    example = {
        'input_ids': np.array(input_ids),
        'attention_mask': np.array(input_mask),
        'labels': np.array(label_ids)
    }
    return example


def process_file (data, tokenizer, number_of_examples, tensor_length=512, label_ignore_id=-100):
    file_examples = []

    for i, entry in tqdm(enumerate(data)):
        if number_of_examples == 0:
            break

        context = entry["context"]
        label = ""
        j = 0

        for qa in entry["qas"]:
            question = qa["question"]
            answer = qa["answers"][0]

            context += '</s> ' + question + f' <extra_id_{j}>'
            label += f' <extra_id_{j}> '+answer
            j+=1

        label += f' <extra_id_{j}>'
        ids = tokenizer(context).input_ids
        label_ids = tokenizer(label).input_ids
        new_example = create_example(ids, label_ids, tensor_length, label_ignore_id)

        if new_example == None:
            continue
        file_examples.append(new_example)
        number_of_examples -= 1

    return file_examples

def main():
    args = get_args()
    print(args)

    output_dir = os.path.split(args.output_path)[0]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    paths = glob(args.input_path)
    examples = []
    max_number_of_examples = args.max_number_of_examples
    for i, path in enumerate(tqdm(paths)):
        print(f'process file {path}')
        data, header = read_mrqa(path)
        tokenizer = tokenization.Tokenizer(args.tokenizer, cache_dir=args.cache_dir)
        file_examples = process_file(data, tokenizer, max_number_of_examples, args.tensor_length, args.label_ignore_id)
        examples += file_examples
        max_number_of_examples -= len(file_examples)

    assert (len(examples) > 0) # file should not be empty
    print(f'saving {len(examples)} examples to {args.output_path}')
    torch.save(examples, args.output_path)
    print('done')


if __name__ == '__main__':
    main()