import tensorflow as tf
import numpy as np

def dense_norm_relu(inputs, units, regularizer=None, is_train=False, name=None, trainable=True):
    a = tf.layers.dense(
        inputs=inputs,
        units=units,
        activation=None,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        kernel_regularizer=regularizer,
        bias_regularizer=regularizer,
        trainable=trainable
        )
    b = tf.layers.batch_normalization(inputs=a, training=is_train, trainable=True)
    return tf.nn.relu(features=b)

def conv_norm_relu(inputs, filters, kernel_size, conv_strides, regularizer=None, is_train=False, trainable=True):
    a = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=conv_strides,
        padding="same",
        activation=None,
        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        kernel_regularizer=regularizer,
        bias_regularizer=regularizer,
        trainable=trainable
        )
    b = tf.layers.batch_normalization(inputs=a, training=is_train, trainable=True)
    return tf.nn.relu(features=b)
'''
def flatten(inputs):
    shape = inputs.get_shape().as_list()
    n = 1
    for s in shape[1:]:
        n *= s
    return tf.reshape(inputs, [-1,n])
'''

def global_avg_pool(inputs):
    a = tf.reduce_mean(inputs, axis=[1,2])
    return flatten(a)