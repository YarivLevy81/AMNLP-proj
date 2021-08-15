# python3 check_trained_model.py --path=/home/yandex/AMNLP2021/sehaik/processed_wiki_val/wiki_split_file_0.tfrecord --checkpoint=/home/yandex/AMNLP2021/sehaik/pretrain_out/model.ckpt-14000

import argparse
import os
from glob import glob
from tqdm import tqdm
import collections

import tensorflow
import numpy as np
tf = tensorflow.compat.v1
#tf.disable_eager_execution()

import tokenization
from transformers import TFT5ForConditionalGeneration, T5Config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    return args


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)


def load_model(checkpoint):
    config = T5Config(pad_token_id=0,  # [PAD]
                      eos_token_id=102,  # [SEP]
                      bos_token_id=101,  # [CLS]
                      decoder_start_token_id=0,  # [PAD]
                      sep_token_id=102,  # [SEP]
                      vocab_size=28996)
    model = TFT5ForConditionalGeneration(config)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    (assignment_map, initialized_variable_names
     ) = get_assignment_map_from_checkpoint(tvars, checkpoint)
    tf.train.init_from_checkpoint(checkpoint, assignment_map)

    return model

def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t

    return example


def read_features(filename, max_seq_length=512):
    name_to_features = dict()
    name_to_features["input_ids"] = tf.FixedLenFeature([max_seq_length], tf.int64)
    name_to_features["input_mask"] = tf.FixedLenFeature([max_seq_length], tf.int64)
    name_to_features["masked_span_ids"] = tf.FixedLenFeature([max_seq_length], tf.int64)

    d = tf.data.TFRecordDataset(tf.constant(filename))
    d = d.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=1,
            num_parallel_batches=1,
            drop_remainder=True))
    return d


def print_output(model, tokenizer, dataset):
    for features in dataset.take(1):
        print('input_tokens:\n', " ".join(tokenizer.convert_ids_to_tokens(features['input_ids'].numpy()[0])))
        print('masked_tokens:\n', " ".join(tokenizer.convert_ids_to_tokens(features['masked_span_ids'].numpy()[0])))

        loss = model(input_ids=features['input_ids'], attention_mask=features['input_mask'],
                        labels=features['masked_span_ids']).loss
        loss = tf.math.reduce_sum(loss)
        print('loss:', loss)

        outputs = model.generate(input_ids=features['input_ids'],
                     attention_mask=features['input_mask'])
        print('output:\n', " ".join(tokenizer.convert_ids_to_tokens(outputs.numpy()[0])))

def main():
    args = get_args()

    tokenizer = tokenization.FullTokenizer(
        vocab_file='/home/yandex/AMNLP2021/sehaik/AMNLP-proj/pretraining/vocabs/bert-cased-vocab.txt',
        do_lower_case=True)
    model = load_model(args.checkpoint)

    paths = glob(args.path)
    for path in tqdm(paths):
        print(f'evaluating file with path: {path}')
        dataset = read_features(path)
        print_output(model, tokenizer, dataset)


if __name__ == '__main__':
    main()