from .nn import nn
import tensorflow as tf
from .layers import *

class vgg(nn):
    def build(self):
        with self.graph.as_default():
            # forward
            with tf.variable_scope("inputs"):
                self.features = tf.placeholder(tf.float32, shape=[None] + self.size)
                self.labels   = tf.placeholder(tf.float32, shape=[None, self.n_lbl])
                self.is_train = tf.placeholder(tf.bool)
                global_step = tf.Variable(0, trainable=False)
            
            with tf.variable_scope("regularization"):
                if self.alpha: regularizer = tf.contrib.layers.l2_regularizer(scale=self.alpha)
                else: regularizer = None
            
            with tf.variable_scope("cnn"):
                conv1 = conv_norm_relu(inputs=self.features, filters=40, kernel_size=[3, 3], conv_strides=[1, 1], regularizer=regularizer, is_train=self.is_train, trainable=True)
                pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=[2, 2])
                conv2 = conv_norm_relu(inputs=pool1, filters=80, kernel_size=[3, 3], conv_strides=[1, 1], regularizer=regularizer, is_train=self.is_train, trainable=True)
                pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=[2, 2])
                conv3 = conv_norm_relu(inputs=pool2, filters=160, kernel_size=[3, 3], conv_strides=[1, 1], regularizer=regularizer, is_train=self.is_train, trainable=True)
                conv4 = conv_norm_relu(inputs=conv3, filters=160, kernel_size=[3, 3], conv_strides=[1, 1], regularizer=regularizer, is_train=self.is_train, trainable=True)
                pool3 = tf.layers.max_pooling2d(conv4, pool_size=[2, 2], strides=[2, 2])
                conv5 = conv_norm_relu(inputs=pool3, filters=320, kernel_size=[3, 3], conv_strides=[1, 1], regularizer=regularizer, is_train=self.is_train, trainable=True)
                conv6 = conv_norm_relu(inputs=conv5, filters=320, kernel_size=[3, 3], conv_strides=[1, 1], regularizer=regularizer, is_train=self.is_train, trainable=True)
                pool4 = tf.layers.max_pooling2d(conv6, pool_size=[2, 2], strides=[2, 2])

            with tf.variable_scope("fc"):
                flat = tf.layers.flatten(inputs=pool4)
                dense1 = dense_norm_relu(inputs=flat, units=1280, regularizer=regularizer, is_train=self.is_train, trainable=True)
                dense2 = dense_norm_relu(inputs=dense1, units=128, regularizer=regularizer, is_train=self.is_train, trainable=True)
                logits = tf.layers.dense(dense2, units=self.n_lbl, activation=None, kernel_regularizer=regularizer, bias_regularizer=regularizer, trainable=True)
            
            with tf.variable_scope("outputs"):
                self.prediction = tf.nn.softmax(logits)
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=logits)) + tf.losses.get_regularization_loss()
            
            
            # backward
            lr = tf.train.exponential_decay(
                learning_rate=self.lr,
                global_step=global_step,
                decay_steps=90*9,
                decay_rate=0.1,
                staircase=True
            )
            optimizer = tf.train.AdamOptimizer(lr)
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                  self.train_op = optimizer.minimize(loss=self.loss)
            
            
            # operation
            self.init_op = tf.global_variables_initializer()