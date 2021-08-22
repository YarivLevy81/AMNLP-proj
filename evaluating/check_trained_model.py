import argparse
import os
from glob import glob
from tqdm import tqdm
import collections

import tensorflow
import numpy as np
tf = tensorflow.compat.v1

from datasets import load_metric
from transformers import TFT5ForConditionalGeneration, AutoConfig

import sys
sys.path.append('../tokenization')
import tokenization

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfrecord_path", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default='t5-base', required=False)
    parser.add_argument("--checkpoint", type=str, default='t5-base', required=False)
    parser.add_argument("--config_name", type=str, default='t5-base', required=False)
    parser.add_argument("--tf_checkpoint", type=str, default="", required=False)
    parser.add_argument("--num_examples", type=int, default=1, required=False)
    parser.add_argument("--max_feature_len", type=int, default=512, required=False)
    parser.add_argument("--min_seq_length", type=int, default=1, required=False)
    parser.add_argument("--cache_dir", type=str, default=None, required=False)
    args = parser.parse_args()
    return args

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


def read_features(filename, max_feature_len=1024):
    name_to_features = dict()
    name_to_features["input_ids"] = tf.FixedLenFeature([max_feature_len], tf.int64)
    name_to_features["input_mask"] = tf.FixedLenFeature([max_feature_len], tf.int64)
    name_to_features["masked_span_ids"] = tf.FixedLenFeature([max_feature_len], tf.int64)

    d = tf.data.TFRecordDataset(tf.constant(filename))
    d = d.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=1,
            num_parallel_batches=1,
            drop_remainder=True))
    return d


def calculate_F1_score(tokenizer, true, pred):
    true = tokenizer.decode(true.numpy()[0], skip_special_tokens=True)
    pred = tokenizer.decode(pred.numpy()[0], skip_special_tokens=True)

    print('true:\n', pred)
    print('pred:\n', pred)

    metric = load_metric("f1")
    metric.add_batch(predictions=pred, references=true)
    score = metric.compute()
    return score


def print_output(model, tokenizer, dataset, max_length = 128, min_length=1, num_examples=1):
    for features in dataset.take(num_examples):
        print('input_tokens:\n', tokenizer.decode(features['input_ids'].numpy()[0]))
        print('masked_tokens:\n', tokenizer.decode(features['masked_span_ids'].numpy()[0]))

        loss = model(input_ids=features['input_ids'], attention_mask=features['input_mask'],
                     labels=features['masked_span_ids']).loss
        print('loss:', loss)
        print('input_mask:', features['input_mask'].numpy())
        loss = tf.math.reduce_sum(loss)
        print('reduced loss:', loss)

        outputs = model.generate(input_ids=features['input_ids'], attention_mask=features['input_mask'],
                                 max_length=max_length, min_length=min_length) # no need for attention mask
        print('output:\n', tokenizer.decode(outputs.numpy()[0]))
        print('output_ids:\n', outputs.numpy()[0])

        print("F1 score:", "not implemented") # FIXME
        #print("F1 score:", calculate_F1_score(tokenizer, features['masked_span_ids'], outputs))


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


def load_model(config_name, checkpoint, tf_checkpoint, cache_dir):
    t5config = AutoConfig.from_pretrained(config_name)
    model = TFT5ForConditionalGeneration(t5config)

    if tf_checkpoint != "":
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        (assignment_map, initialized_variable_names
         ) = get_assignment_map_from_checkpoint(tvars, tf_checkpoint)
        tf.train.init_from_checkpoint(tf_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)
    else:
        model = TFT5ForConditionalGeneration.from_pretrained(checkpoint, cache_dir=cache_dir)

    return model


def main():
    args = get_args()

    tokenizer = tokenization.Tokenizer(args.tokenizer, cache_dir=args.cache_dir)
    model = load_model(args.config_name, args.checkpoint, args.tf_checkpoint, args.cache_dir)

    paths = glob(args.tfrecord_path)
    print('num of paths:', len(paths))
    for path in tqdm(paths):
        print(f'evaluating file with path: {path}')
        dataset = read_features(path, args.max_feature_len)
        print_output(model, tokenizer, dataset, args.max_feature_len, args.min_seq_length, args.num_examples)


if __name__ == '__main__':
    main()