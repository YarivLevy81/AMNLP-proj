import argparse
import os
from tqdm import tqdm
from glob import glob
from multiprocessing import Pool, Manager, Value

import tensorflow.compat.v1 as tf
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default='data', required=False)
    parser.add_argument("--num_workers", type=int, default=16, required=False)
    args = parser.parse_args()
    return args


def get_start_idx(lock, global_idx, size):
    with lock:
        start_idx = global_idx.value
        global_idx.value += size
    return start_idx


def unpack(lock, global_idx, output_dir, prefix, filepath):
    if not filepath.endswith("pt"):
        return
    data = torch.load(filepath)
    size = len(data)
    idx = get_start_idx(lock, global_idx, size)
    tf.print(f'{filepath}: found {size} examples, starting idx = {idx}')

    for e in data:
        filename = os.path.join(output_dir, f'{prefix}_{idx}.pt')
        torch.save(e, filename)
        idx += 1


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    m = Manager()
    global_idx = m.Value('global_idx', 0)
    lock = m.Lock()
    paths = glob(args.input_path)
    params = [(lock, global_idx, args.output_dir, args.prefix, path) for path in paths]
    with Pool(args.num_workers) as p:
        p.starmap(unpack, params)
    tf.print(f'done. globalidx={global_idx.value}')

if __name__ == '__main__':
    args = get_args()
    tf.print(args)
    main(args)