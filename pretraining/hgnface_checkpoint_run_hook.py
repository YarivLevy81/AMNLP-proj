from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow
import datetime
tf = tensorflow.compat.v1

class HGNFCheckpointHook(tf.train.SessionRunHook):
    """Logs model stats to a csv."""

    def __init__(self, model, path):
        self.model = model
        self.path = path

    def save_ckpt(self):
        tf.logging.info("***** Saving huggingface checkpoint *****")
        suffix = datetime.datetime.now()
        self.model.save_pretrained(os.path.join(self.path, str(suffix)))
        tf.logging.info("***** Saving huggingface checkpoint - Done! *****")

    def begin(self):
        self.save_ckpt()

    def end(self, session=None):
        self.save_ckpt()

    def after_run(self, run_context, run_values):
        self.save_ckpt()
