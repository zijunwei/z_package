import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

# spp layer
def spp_layer(input_, levels=None, scope='spp_layer', reuse=None):
    if not levels:
        levels = [1, 2, 3]
    shape = input_.get_shape().as_list()
    with tf.variable_scope(name_or_scope=scope, reuse=reuse):
        pool_outputs = []
        for l in levels:
            pool = tf.nn.avg_pool(input_, ksize=[1, np.ceil(shape[1] * 1. / l).astype(np.int32),
                                                 np.ceil(shape[2] * 1. /l).astype(np.int32), 1],
                                  strides=[1, np.ceil(shape[1] * 1. / l).astype(np.int32),
                                           np.ceil(shape[2] * 1. /l), 1],
                                  padding='SAME')
            pool_outputs.append(tf.reshape(pool, [shape[0], -1]))
        spp_pool = tf.concat(axis=1, values=pool_outputs)
    return spp_pool


# spp layer in visual finding
def spp_vfn(input_, levels=None, scope='spp_vfn_layer', reuse=None):
    if not levels:
        levels = [3, 5, 7]
    shape = input_.get_shape().as_list()
    with tf.variable_scope(name_or_scope=scope, reuse=reuse):
        pool_outputs = []
        for l in levels:
            pool = tf.nn.max_pool(input_, ksize=[1, l, l, 1],
                                  strides=[1, l-1, l-1, 1],
                                  padding='VALID')
            pool_outputs.append(tf.reshape(pool, [shape[0], -1, shape[3]]))
        spp_pool = tf.concat(axis=1, values=pool_outputs)
        spp_pool = tf.reduce_max(spp_pool, reduction_indices=[1])
        tf_reduction = slim.fully_connected(spp_pool, 1000, activation_fn=None,
                                        normalizer_fn=None, scope='spp_reduction', reuse=False,
                                            weights_regularizer=slim.l2_regularizer(0.0001))
    return tf_reduction