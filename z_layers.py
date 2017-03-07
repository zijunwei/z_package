import tensorflow as tf
import numpy as np


# spp layer
def spp_layer(input_, levels=None, scope='SPP_layer', reuse=None):
    if not levels:
        levels = [1, 2, 3]
    shape = input_.get_shape().as_list()
    with tf.variable_scope(name_or_scope=scope, reuse=reuse):
        pool_outputs = []
        for l in levels:
            pool = tf.nn.max_pool(input_, ksize=[1, np.ceil(shape[1] * 1. / l).astype(np.int32),
                                                 np.ceil(shape[2] * 1. /l).astype(np.int32), 1],
                                  strides=[1, np.floor(shape[1] * 1. / l + 1).astype(np.int32),
                                           np.floor(shape[2] * 1. /l + 1), 1],
                                  padding='SAME')
            pool_outputs.append(tf.reshape(pool, [shape[0], -1]))
        spp_pool = tf.concat(axis=1, values=pool_outputs)
    return spp_pool