import argparse
import json
import os
from glob import glob
from tqdm import tqdm

import tensorflow
tf = tensorflow.compat.v1

import tokenization
import collections

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
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


def process_file (data, tokenizer, path):
    for i, entry in tqdm(enumerate(data)):
        context = entry["context"]
        context_tokens = entry["context_tokens"]

        tokens_text = " ".join(x[0] for x in entry["context_tokens"])
        tokens = tokenizer.tokenize(tokens_text)
        label = []
        j = 0
        for qa in entry["qas"]:
            question_text = " ".join(x[0] for x in qa["question_tokens"])
            question_tokens = tokenizer.tokenize(question_text)

            answer = qa["detected_answers"][0]
            answer_text = " ".join(
                [c_t[0] for c_t in context_tokens[answer['token_spans'][0][0]: answer['token_spans'][0][1] + 1]])

            tokens = tokens + ['.'] + question_tokens + [f"[unused{j+1}]"]
            label = label + [f"[unused{j+1}]"] + tokenizer.tokenize(answer_text) # FIXME - add extra?

        example_path =  os.path.splitext(path)[0] + f"_{i}.tfrecord"
        save_tfrecords(tokens, label, example_path, tokenizer)
        #print('tokens', tokens)
        #print('label', label)
        #exit(0)


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature

def save_tfrecords(tokens, label, example_path, tokenizer, max_length=512):

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_len = len(input_ids)
    input_mask = [1] * input_len
    assert input_len <= max_length

    while len(input_ids) < max_length:
        input_ids.append(0)
        input_mask.append(0)

    assert len(input_ids) == max_length
    assert len(input_mask) == max_length

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)

    # masked_span_positions = list(instance.masked_span_positions) # FIXME ?
    masked_span_ids = tokenizer.convert_tokens_to_ids(label)
    masked_span_ids += [0] * (max_length - len(masked_span_ids))

    assert len(masked_span_ids) == max_length

    #features["masked_span_positions"] = create_int_feature(masked_span_positions)
    features["masked_span_ids"] = create_int_feature(masked_span_ids)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    with tf.io.TFRecordWriter(example_path) as file_writer:
        file_writer.write(tf_example.SerializeToString())


def main():
    args = get_args()
    paths = glob(args.path)
    for path in tqdm(paths):
        data, header = read_mrqa(path)
        tokenizer = tokenization.FullTokenizer(
            vocab_file='/home/yandex/AMNLP2021/sehaik/AMNLP-proj/pretraining/vocabs/bert-cased-vocab.txt', do_lower_case=True)
        process_file(data, tokenizer, path)


if __name__ == '__main__':
    main()