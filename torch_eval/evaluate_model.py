import argparse
import os
from glob import glob
from tqdm import tqdm

from transformers import TFT5ForConditionalGeneration, AutoConfig
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
    parser.add_argument("--max_seq_length", type=int, default=128, required=False)
    parser.add_argument("--min_seq_length", type=int, default=1, required=False)
    parser.add_argument("--cache_dir", type=str, default=None, required=False)
    args = parser.parse_args()
    return args


def calculate_F1_score(label_text, generated_text):
    label = set(label_text.split())
    generated = set(generated_text.split())
    common = len(label.intersect(generated))
    precision = common / generated
    recall = common / label
    if (recall + precision):
        return 2.0 * (precision * recall) / (recall + precision)
    return 0


def calculate_metrics(args, model, tokenizer, dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    total_score, count = 0.0, 0.0
    for i, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                                 max_length=args.max_seq_length, min_length=args.min_seq_length)
        label = batch['labels'][0]
        label_text = tokenizer.decode(label, skip_special_tokens=True)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        f1_score = calculate_F1_score(label_text, generated_text)
        print(f'label: {label_text}')
        print(f'generates: {generated_text}')
        print(f'F1 score: {f1_score}')
        total_score += f1_score
        count += 1
    return total_score / count


def create_dataloader(dataset_path):
    dataset = splinter_dataset.SplinterDataset(dataset_path, train=False)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    return dataloader


def load_checkpoint(checkpoint_path, model):
    print('Loading from checkpoint')
    ckpt = torch.load(checkpoint_path)
    model.load_state_dict(ckpt['model'])


def eval(args, tokenizer, model):
    print(f'evaluating file with path: {path}')
    dataloader = create_dataloader(path)
    metrics = calculate_metrics(args, model, tokenizer, dataloader)
    return metrics


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.config, cache_dir=args.cache_dir)
    if args.from_pretrained:
        model = T5ForConditionalGeneration.from_pretrained(args.config, cache_dir=args.cache_dir)
    else:
        t5config = AutoConfig.from_pretrained(args.config, cache_dir=args.cache_dir)
        model = T5ForConditionalGeneration(t5config)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model)

    paths = glob(args.tfrecord_path)
    total_score, count, avg_score = 0.0 , 0.0, 0.0
    print('num of paths:', len(paths))
    for path in tqdm(paths):
        f1_score = eval(path, tokenizer, model)
        total_score += f1_score
        count += 1
    if count:
        avg_score = total_score / count
    print(f'Average F1 score for files: {avg_score}')


if __name__ == '__main__':
    args = get_args()
    main(args)