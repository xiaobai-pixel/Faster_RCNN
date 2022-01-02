from __future__ import print_function

import tensorflow as tf
import numpy as np

from roi_pooling.roi_pooling_ops import roi_pooling



# 4x4 feature map with only 1 channel
input_value = [[
    [[1], [2], [4], [4]],
    [[3], [4], [1], [2]],
    [[6], [2], [1], [7]],
    [[1], [3], [2], [8]]
]]
input_value = np.asarray(input_value, dtype='float32')

# regions of interest as lists of:
# feature map index, upper left, bottom right coordinates
rois_value = [
    [0, 0, 0, 1, 3],
    [0, 2, 2, 3, 3],
    [0, 1, 0, 3, 2]
]
rois_value = np.asarray(rois_value, dtype='int32')

# in this case we have 3 RoI pooling operations:
# * channel 0, rectangular region (0, 0) to (1, 3)
#              xx..
#              xx..
#              xx..
#              xx..
#
# * channel 0, rectangular region (2, 2) to (3, 3)
#              ....
#              ....
#              ..xx
#              ..xx
# * channel 0, rectangular region (1, 0) to (3, 2)
#              ....
#              xxx.
#              xxx.
#              xxx.

input_featuremap = tf.placeholder(tf.float32)
rois = tf.placeholder(tf.int32)
input_const = tf.constant(input_value, tf.float32)
rois_const = tf.constant(rois_value, tf.int32)
y = roi_pooling(input_const, rois_const, pool_height=2, pool_width=2)

with tf.Session('') as sess:
    y_output = sess.run(y, feed_dict={input_featuremap: input_value, rois: rois_value})
    print(y_output)


#122
#1112
#11221