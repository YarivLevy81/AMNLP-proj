import sys
import tensorflow
tf = tensorflow.compat.v1

# This example supposes that the events file contains summaries with a
# summary value tag 'loss'.  These could have been added by calling
# `add_summary()`, passing the output of a scalar summary op created with
# with: `tf.scalar_summary(['loss'], loss_tensor)`.
for e in tf.train.summary_iterator(sys.argv[1]):
    for v in e.summary.value:
        if v.tag == 'loss':
            print(v.simple_value)
