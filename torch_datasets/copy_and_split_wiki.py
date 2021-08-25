import sys
import os
import argparse
from glob import glob
from tqdm import tqdm
import ntpath

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default='/home/yandex/AMNLP2021/data/wiki/all', required=False)
    parser.add_argument("--output_dir", type=str, default='/home/yandex/AMNLP2021/sehaik/wiki_split_3/', required=False)
    parser.add_argument("--num_bytes", type=int, default=20000000, required=False)
    args = parser.parse_args()
    return args


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def split_file(path, output_dir, num_bytes):
    cur_byte = 0
    count = 0
    fsize = os.path.getsize(path)
    file = open(path, "r")

    while True:
        lines = file.read(num_bytes)
        res_path = os.path.join(output_dir, f"file_{count}")

        with open(res_path, "w") as f:
            f.write(lines)
        print(f"saved {res_path}. cur_byte={cur_byte}")

        cur_byte += num_bytes
        if (cur_byte > fsize):
            break
        count += 1

    print(f"done with file {path}")
    file.close()


def main():
    args = get_args()
    print(f'split file:\n{args}')

    paths = glob(args.input_path)
    for path in tqdm(paths):
        print(f'splitting file: {path}')
        cur_dir = os.path.join(args.output_dir, path_leaf(path))
        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)
        print("Directory '%s' created" % cur_dir)
        split_file(path, cur_dir, args.num_bytes)


if __name__ == '__main__':
    main()
