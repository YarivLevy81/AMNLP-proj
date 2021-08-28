import argparse
import os
from glob import glob
from tqdm import tqdm
from collections import Counter
import string
import re

import tensorflow.compat.v1 as tf
import torch

from transformers import T5ForConditionalGeneration, AutoConfig
from transformers import AutoTokenizer


import sys
sys.path.append('../tokenization')
sys.path.append('../torch_datasets')
import tokenization
import splinter_dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--config", type=str, default='t5-small', required=False)
    parser.add_argument("--checkpoint", type=str, default=None, required=False)
    parser.add_argument("--from_pretrained", type=bool, default=False, required=False)
    parser.add_argument("--cache_dir", type=str, default=None, required=False)
    args = parser.parse_args()
    return args


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


# note that in the real mrqa eval script we take the max over ground truths (i.e. best),
# and here we compare to some ground truth
# code was taken from mrqa github
def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def calculate_metrics(args, model, tokenizer, dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    total_score, total, exact_match = 0.0, 0.0, 0.0
    for i, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        label = [x for x in batch['labels'][0] if x!=-100]
        label_text = tokenizer.decode(label, skip_special_tokens=True)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        exact_match += exact_match_score(label_text, generated_text)
        score = f1_score(label_text, generated_text)
        tf.print(f'Label: {label_text}')
        tf.print(f'Generated: {generated_text}')
        tf.print(f'F1 score: {score}')
        total_score += score
        total += 1
    return total_score, exact_match, total


def create_dataloader(dataset_path):
    dataset = splinter_dataset.SplinterDataset(dataset_path)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    tf.print(f'Created dataset ({len(dataset)} examples)')
    return dataloader


def load_checkpoint(checkpoint_path, model):
    tf.print('Loading from checkpoint')
    ckpt = torch.load(checkpoint_path)
    model.load_state_dict(ckpt['model'])


def eval(args, tokenizer, model, path):
    tf.print(f'Evaluating file with path: {path}')
    dataloader = create_dataloader(path)
    metrics = calculate_metrics(args, model, tokenizer, dataloader)
    return metrics


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.config, cache_dir=args.cache_dir)
    if args.from_pretrained:
        tf.print(f'Loading pretrained: {args.config}')
        model = T5ForConditionalGeneration.from_pretrained(args.config, cache_dir=args.cache_dir)
    else:
        tf.print(f'Initializing random: {args.config}')
        t5config = AutoConfig.from_pretrained(args.config, cache_dir=args.cache_dir)
        model = T5ForConditionalGeneration(t5config)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model)

    paths = glob(args.input_path)
    total_score, exact_match_total, total, avg_score, avg_match = 0.0 , 0.0, 0.0, 0.0, 0.0
    tf.print('num of paths:', len(paths))
    for path in tqdm(paths):
        f1_score, exact_match, count = eval(path, tokenizer, model, path)
        total_score += f1_score
        exact_match_total += exact_match
        total += count
    if total:
        avg_score = 100 * total_score / total
        avg_match = 100 * exact_match_total / total
    tf.print(f'\nAverage F1 score for files: {avg_score}')
    tf.print(f'Average of exact matches for files: {avg_match}')


if __name__ == '__main__':
    tf.print('Started')
    args = get_args()
    tf.print(args)
    main(args)