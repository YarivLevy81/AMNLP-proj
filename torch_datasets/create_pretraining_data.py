# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create masked LM/next sentence masked_lm TF examples for BERT."""
import functools
import os
import collections
import random
import time
from multiprocessing import Pool
import pickle
import numpy as np
import torch

import tensorflow.compat.v1 as tf

from masking import create_recurring_span_selection_predictions
import sys
sys.path.append('../tokenization')
import tokenization

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_file", None,
    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string(
    "output_dir", None,
    "Output TF example directory.")

flags.DEFINE_integer(
    "max_questions_per_seq", 15,
    "")

flags.DEFINE_integer(
    "max_seq_length", 256,
    "Maximum sequence length.")

flags.DEFINE_integer(
    "max_feature_length", 512,
    "Maximum length of feature.")

flags.DEFINE_integer(
    "random_seed", 12345,
    "Random seed for data generation.")

flags.DEFINE_integer(
    "num_processes", 63,
    "Number of processes")

flags.DEFINE_integer(
    "dupe_factor", 10,
    "Number of times to duplicate the input data (with different masks).")

flags.DEFINE_float(
    "masked_lm_prob", 0.15,
    "Masked LM probability.")

flags.DEFINE_integer(
    "max_span_length", 10,
    "Maximum span length to mask")

flags.DEFINE_bool(
    "verbose", False,
    "verbose")

flags.DEFINE_string(
    "tokenizer", 't5-small',
    "T5 model size.")

flags.DEFINE_string(
    "cache_dir", None,
    "cache dir for pretrained tokenizer")


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, tokens, masked_span_tokens=None):
        self.tokens = tokens
        self.masked_span_tokens = masked_span_tokens
        print(self.__str__())

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(self.tokens))
        s += "masked_span_tokens: %s\n" % (" ".join(self.masked_span_tokens))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


class DataStatistics:
    def __init__(self):
        self.num_contexts = 0
        self.ngrams = collections.Counter()


def write_instance_to_example_files(instances, tokenizer, tensor_length, output_files):
    """Create TF example files from `TrainingInstance`s."""
    total_written, total_tokens_written, skipped_instances = 0, 0, 0
    examples = []
    for (inst_index, instance) in enumerate(instances):
        input_ids = tokenizer(" ".join(instance.tokens)).input_ids
        input_len = len(input_ids)
        input_mask = [1 if input_ids[i] != 0 else 0 for i in range(len(input_ids))]
        masked_span_ids = tokenizer(" ".join(instance.masked_span_tokens)).input_ids
        masked_span_ids = [-100] + masked_span_ids

        if input_len > tensor_length or input_len != len(input_mask) or \
                len(masked_span_ids) > tensor_length:
            print("skipping...")
            skipped_instances += 1
            continue

        total_written += 1
        total_tokens_written += input_len

        input_ids += [0] * (tensor_length - len(input_ids))
        input_mask += [0] * (tensor_length - len(input_mask))
        masked_span_ids += [-100] * (tensor_length - len(masked_span_ids))

        assert len(input_ids) == tensor_length
        assert len(input_mask) == tensor_length
        assert len(masked_span_ids) == tensor_length

        example = {
            'input_ids': np.array(input_ids),
            'attention_mask': np.array(input_mask),
            'labels': np.array(masked_span_ids)
        }
        examples.append(example)

    torch.save(examples, output_files[0])
    if total_written != 0:
        print(f"Skipped {skipped_instances} instances")
        print(
            f"Wrote {total_written} total instances, average length is {total_tokens_written // total_written}")


def create_training_instances(input_file, tokenizer, max_seq_length, dupe_factor, masked_lm_prob,
                              rng, max_questions_per_seq, max_span_length, statistics=None):
    """Create `TrainingInstance`s from raw text."""
    all_documents = [[]]

    # Wiki Input file format:
    # (1) <doc> starts a new wiki document and </doc> ends it
    # (2) One sentence/paragraph per line.
    # BookCorpus file format:
    # Empty line starts a new book
    with tf.gfile.GFile(input_file, "r") as reader:
        for i, line in enumerate(reader):
            line = tokenizer._clean_text(line)
            if (not line) or line.startswith("</doc"):
                continue
            if line.startswith("<doc"):
                all_documents.append([])
            tokens = line.split()
            if tokens:
                all_documents[-1].append(tokens)

    # Remove empty documents
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)

    instances = []
    for dupe_idx in range(dupe_factor):
        for document_index in range(len(all_documents)):
            instances.extend(
                create_instances_from_document(
                    all_documents, document_index, max_seq_length,
                    dupe_idx, masked_lm_prob, max_questions_per_seq,
                    max_span_length, statistics))

    rng.shuffle(instances)
    return instances


def create_instance_from_context(segments, max_span_length, masked_lm_prob,
                                 max_questions_per_seq, statistics=None):
    tokens = []
    for segment in segments:
        tokens += segment

    tokens, span_label_tokens, span_clusters = \
        create_recurring_span_selection_predictions(tokens,
                                                    max_questions_per_seq,
                                                    max_span_length,
                                                    masked_lm_prob)
    if tokens is None:
        return None
    statistics.ngrams.update([cluster[1] for cluster in span_clusters])

    statistics.num_contexts += 1

    return TrainingInstance(tokens=tokens,
                            masked_span_tokens=span_label_tokens)


def create_instances_from_document(all_documents, document_index, max_seq_length, dupe_idx,
                                   masked_lm_prob, max_questions_per_seq, max_span_length, statistics=None):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[document_index]

    # Account for <pad> </s>
    max_num_tokens = max_seq_length

    instances = []
    current_chunk = []
    current_length = 0
    for i, segment in enumerate(document):
        segment_len = len(segment)

        if current_length + segment_len > max_num_tokens or (len(document) >= 10 and i % len(document) == dupe_idx):
            if current_chunk:
                current_chunk.append(segment[:max_num_tokens - current_length])
                instance = create_instance_from_context(current_chunk, max_span_length, masked_lm_prob,
                                                        max_questions_per_seq, statistics)
                if instance is None:
                    continue
                instances.append(instance)

            current_chunk, current_length = [], 0
            if segment_len > max_num_tokens:
                # If this segment is too long, take the first max_num_tokens from this segment
                segment = segment[:max_num_tokens]

        current_chunk.append(segment)
        current_length += len(segment)

    if current_chunk:
        instance = create_instance_from_context(current_chunk, max_span_length, masked_lm_prob,
                                                max_questions_per_seq, statistics)
        if instance is not None:
            instances.append(instance)

    return instances


def process_file(input_file, output_file, tokenizer, rng):
    print(f"*** Started processing file {input_file} ***")

    statistics = DataStatistics()

    instances = create_training_instances(
        input_file, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
        FLAGS.masked_lm_prob, rng, FLAGS.max_questions_per_seq,
        FLAGS.max_span_length, statistics)

    print(f"*** Finished processing {statistics.num_contexts} contexts from file {input_file}, "
                    f"writing to output file {output_file} ***")

    write_instance_to_example_files(instances, tokenizer, FLAGS.max_feature_length, [output_file])

    print(f"*** Finished writing to output file {output_file} ***")
    return statistics


def get_output_file(input_file, output_dir):
    path = os.path.normpath(input_file)
    split = path.split(os.sep)
    dir_and_file = split[-2:]
    return os.path.join(output_dir, '_'.join(dir_and_file) + '.pt')


def main(_):

    assert not tf.io.gfile.exists(FLAGS.output_dir), "Output directory already exists"
    tf.io.gfile.mkdir(FLAGS.output_dir)

    tokenizer = tokenization.Tokenizer(FLAGS.tokenizer, cache_dir=FLAGS.cache_dir)

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))

    print(f"*** Reading from {len(input_files)} files ***")

    rng = random.Random(FLAGS.random_seed)

    params = [(file, get_output_file(file, FLAGS.output_dir), tokenizer, rng)
              for file in input_files]
    with Pool(FLAGS.num_processes if FLAGS.num_processes else None) as p:
        results = p.starmap(process_file, params)

    total_num_contexts = sum([res.num_contexts for res in results])

    print("Finished writing all files! Aggregating statistics..")
    ngrams = functools.reduce(lambda x, y: x + y, [res.ngrams for res in results])
    ngrams = collections.Counter({" ".join(tokens): num for tokens, num in ngrams.items()}).most_common()
    print(f"100 most common ngrams: {ngrams[:100]}")

    ngrams_file = os.path.join(FLAGS.output_dir, "ngrams.txt")
    with tf.gfile.GFile(ngrams_file, "w") as writer:
        for ngram, num in ngrams:
            writer.write(f"{ngram}\t{num}\n")
    print(f"Number of unique n-grams: {len(ngrams)}")

    print(f"Done! Total number of contexts: {total_num_contexts}")


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
