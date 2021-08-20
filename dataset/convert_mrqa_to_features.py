import argparse
import json
import os
from glob import glob
from tqdm import tqdm

import tensorflow
tf = tensorflow.compat.v1

import collections
import ntpath

import sys
sys.path.append('../tokenization')
import tokenization


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default='t5-base', required=False)
    parser.add_argument("--cache_dir", type=str, default=None, required=False)
    parser.add_argument("--max_feature_length", type=int, default=512, required=False)
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


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature

def save_tfrecords(input_ids, label_ids, example_path, max_feature_length=512):

    input_len = len(input_ids)
    input_mask = [1 if input_ids[i] != 0 else 0 for i in range(len(input_ids))]

    if input_len > max_feature_length or input_len != len(input_mask) or \
            len(label_ids) > max_feature_length:
        print("skipping...")
        return

    input_ids += [0] * (max_feature_length - len(input_ids))
    input_mask += [0] * (max_feature_length - len(input_mask))
    label_ids += [0] * (max_feature_length - len(label_ids))

    assert len(input_ids) == max_feature_length
    assert len(input_mask) == max_feature_length
    assert len(label_ids) == max_feature_length

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["masked_span_ids"] = create_int_feature(label_ids)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    with tf.io.TFRecordWriter(example_path) as file_writer:
        file_writer.write(tf_example.SerializeToString())


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def process_file (data, tokenizer, path, output_dir, max_feature_length=512):
    global len_c
    global len_l
    for i, entry in tqdm(enumerate(data)):
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

        #print('context:', context)
        #print('label:', label)
        #print('ids:', ids)
        #print('label_ids:', label_ids)

        f_name = os.path.splitext(path_leaf(path))[0]
        example_path =  os.path.join(output_dir, f'{f_name}_{i}.tfrecord')
        print('saved file: ', example_path)
        save_tfrecords(ids, label_ids, example_path, max_feature_length)


def main():
    args = get_args()
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    paths = glob(args.input_path)
    for path in tqdm(paths):
        data, header = read_mrqa(path)
        print('tokenizer:', args.tokenizer)
        tokenizer = tokenization.Tokenizer(args.tokenizer, cache_dir=args.cache_dir)
        process_file(data, tokenizer, path, output_dir, args.max_feature_length)


if __name__ == '__main__':
    main()