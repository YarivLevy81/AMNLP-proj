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
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow
tf = tensorflow.compat.v1

from transformers import TFT5ForConditionalGeneration, AutoConfig
from optimization import train_op_fn
from hgnface_checkpoint_run_hook import HGNFCheckpointHook

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "train_input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "validation_input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "checkpoint", 't5-base',
    "Initial checkpoint")

flags.DEFINE_string(
    "cache_dir", None,
    "Initial checkpoint")

## Other parameters
flags.DEFINE_string(
    "config", 't5-base',
    "T5 config")

flags.DEFINE_integer(
    "max_feature_length", 512,
    "The maximum length of a feature")

flags.DEFINE_integer(
    "train_batch_size", 4,
    "Total batch size for training.")

flags.DEFINE_integer(
    "gradient_accumulation_multiplier", 8,
    "backpropagation every gradient_accumulation_multiplier steps")

# REMARK: effective training batch size = train_batch_size*gradient_accumulation_multiplier

flags.DEFINE_integer(
    "eval_batch_size", 4,
    "Total batch size for eval.")

flags.DEFINE_float(
    "learning_rate", 1e-4,
    "The initial learning rate for Adam.")

flags.DEFINE_integer(
    "num_train_steps", 1000000,
    "Number of training steps.")

flags.DEFINE_integer(
    "num_warmup_steps", 10000,
    "Number of warmup steps.")

flags.DEFINE_integer(
    "save_checkpoints_steps", 10000,
    "How often to save the model checkpoint.")

flags.DEFINE_integer(
    "keep_checkpoint_max", 100,
    "maximum number of checkpoint to keep")

flags.DEFINE_integer(
    "iterations_per_loop", 1000,
    "How many steps to make in each estimator call.")

#flags.DEFINE_integer(
#    "max_eval_steps", 100,
#    "Maximum number of eval steps.")

def model_fn_builder(checkpoint, cache_dir, config_name, init_lr, gradient_accumulation_multiplier, num_train_steps,
                            num_warmup_steps, output_dir, save_summary_steps):
    """Returns `model_fn` closure for TFEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TFEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        masked_span_ids = features["masked_span_ids"]

        t5config = AutoConfig.from_pretrained(config_name)
        model = TFT5ForConditionalGeneration(t5config)

        if checkpoint:
            model = TFT5ForConditionalGeneration.from_pretrained(checkpoint,cache_dir=cache_dir)

        # the forward function automatically creates the correct decoder_input_ids
        total_loss = model(input_ids=input_ids, labels=masked_span_ids, attention_mask=input_mask).loss
        total_loss = tf.math.reduce_sum(total_loss)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            summary_hook = tf.train.SummarySaverHook(
                save_steps=save_summary_steps,
                output_dir=os.path.join(output_dir, "train"),
                summary_op=tf.summary.scalar('loss', total_loss))

            save_checkpoint_hook = HGNFCheckpointHook(
                model=model,
                path=os.path.join(output_dir, "hf_ckpt")
            )

            train_op = train_op_fn(
                total_loss,
                init_lr,
                num_train_steps,
                num_warmup_steps,
                gradient_accumulation_multiplier
            )

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                training_hooks=[summary_hook, save_checkpoint_hook])
        elif mode == tf.estimator.ModeKeys.EVAL:
            summary_hook = tf.train.SummarySaverHook(
                save_steps=1, # TODO check
                output_dir=os.path.join(output_dir, "eval"),
                summary_op=tf.summary.scalar('loss', total_loss))

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                #eval_metric_ops={}, # this is usefull for F1 score
                evaluation_hooks=[summary_hook])
        else:
            raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

        return output_spec

    return model_fn


def input_fn_builder(input_files,
                     max_feature_length,
                     is_training,
                     batch_size,
                     num_cpu_threads=4):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn(params):
        """The actual input function."""

        name_to_features = dict()
        name_to_features["input_ids"] = tf.FixedLenFeature([max_feature_length], tf.int64)
        name_to_features["input_mask"] = tf.FixedLenFeature([max_feature_length], tf.int64)
        name_to_features["masked_span_ids"] = tf.FixedLenFeature([max_feature_length], tf.int64)

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
            d = d.repeat()
            d = d.shuffle(buffer_size=len(input_files))

            # `cycle_length` is the number of parallel files that get read.
            cycle_length = min(num_cpu_threads, len(input_files))

            # `sloppy` mode means that the interleaving is not exact. This adds
            # even more randomness to the training pipeline.
            d = d.apply(
                tf.data.experimental.parallel_interleave(
                    tf.data.TFRecordDataset,
                    sloppy=is_training,
                    cycle_length=cycle_length))
            d = d.shuffle(buffer_size=100)
        else:
            # For eval we want out-of-range exceptions.
            d = tf.data.TFRecordDataset(tf.constant(input_files))

        # We must `drop_remainder` on training because the TPU requires fixed
        # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
        # and we *don't* want to drop the remainder, otherwise we wont cover
        # every sample.
        d = d.apply(
            tf.data.experimental.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                num_parallel_batches=num_cpu_threads,
                drop_remainder=True))
        return d

    return input_fn


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


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.gfile.MakeDirs(FLAGS.output_dir)

    train_input_files = []
    for input_pattern in FLAGS.train_input_file.split(","):
        train_input_files.extend(tf.gfile.Glob(input_pattern))

    val_input_files = []
    for input_pattern in FLAGS.validation_input_file.split(","):
        val_input_files.extend(tf.gfile.Glob(input_pattern))


    tf.logging.info("*** Train Input Files ***")
    for input_file in train_input_files:
        tf.logging.info("  %s" % input_file)

    tf.logging.info("*** Validation Input Files ***")
    for input_file in val_input_files:
        tf.logging.info("  %s" % input_file)

    run_config = tf.estimator.RunConfig(
        output_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        keep_checkpoint_max=FLAGS.keep_checkpoint_max,
        iterations_per_loop=FLAGS.iterations_per_loop
    )

    model_fn = model_fn_builder(
        checkpoint=FLAGS.checkpoint,
        cache_dir=FLAGS.cache_dir,
        config_name=FLAGS.config,
        init_lr=FLAGS.learning_rate,
        gradient_accumulation_multiplier=FLAGS.gradient_accumulation_multiplier,
        num_train_steps=FLAGS.num_train_steps,
        num_warmup_steps=FLAGS.num_warmup_steps,
        output_dir=FLAGS.output_dir,
        save_summary_steps=FLAGS.save_checkpoints_steps) # this is on purpose save_checkpoints_steps

    # Normal Estimator on CPU or GPU.
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config)

    train_input_fn = input_fn_builder(
        input_files=train_input_files,
        batch_size=FLAGS.train_batch_size,
        max_feature_length=FLAGS.max_feature_length,
        is_training=True)

    eval_input_fn = input_fn_builder(
        input_files=val_input_files,
        batch_size=FLAGS.train_batch_size, # this is on purpose train_batch_size
        max_feature_length=FLAGS.max_feature_length,
        is_training=False)

    # setup train spec
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=FLAGS.num_train_steps)

    # setup eval spec evaluating every checkpoint
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)

    # run train and evaluate
    result = tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    print('result:', result)


if __name__ == "__main__":
    flags.mark_flag_as_required("train_input_file")
    flags.mark_flag_as_required("validation_input_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
