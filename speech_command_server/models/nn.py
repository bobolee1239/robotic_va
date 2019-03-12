import tensorflow as tf
import numpy as np
from .layers import *

class nn(object):
    def __init__(self, size, n_lbl, lr=1e-3, dr=False, alpha=False):
        # parameters
        self.size  = size
        self.n_lbl = n_lbl
        self.lr    = lr
        self.dr    = dr
        self.alpha = alpha
        
        
        # operation
        self.graph = tf.Graph()
        self.build()
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init_op)
    
    def build(self):
        with self.graph.as_default():
            # forward
            with tf.variable_scope("inputs"):
                self.features = tf.placeholder(tf.float32, shape=[None] + self.size)
                self.labels   = tf.placeholder(tf.float32, shape=[None, self.n_lbl])
                self.is_train = tf.placeholder(tf.bool)
                global_step = tf.Variable(0, trainable=False)
                regularizer = None
                
            with tf.variable_scope("fc"):
                dense1 = dense_norm_relu(inputs=self.features, units=1024, regularizer=regularizer, is_train=self.is_train, trainable=True)
                dense2 = dense_norm_relu(inputs=dense1, units=512, regularizer=regularizer, is_train=self.is_train, trainable=True)
                dense3 = dense_norm_relu(inputs=dense2, units=256, regularizer=regularizer, is_train=self.is_train, trainable=True)
                logits = tf.layers.dense(dense3, units=self.n_lbl, activation=None, kernel_regularizer=regularizer, bias_regularizer=regularizer, trainable=True)
                
            with tf.variable_scope("outputs"):
                # regression
                self.prediction = logits
                self.loss = tf.losses.mean_squared_error(self.labels, self.prediction) + tf.losses.get_regularization_loss() 
                
                # classification
                '''
                self.prediction = tf.nn.softmax(logits)
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=logits)) + tf.losses.get_regularization_loss()
                '''
                
            # backward
            lr = tf.train.exponential_decay(
                learning_rate=self.lr,
                global_step=global_step,
                decay_steps=700*5,
                decay_rate=0.1,
                staircase=True
            )
            optimizer = tf.train.AdamOptimizer(lr)
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                  self.train_op = optimizer.minimize(loss=self.loss)

            # operation
            self.init_op = tf.global_variables_initializer()
    
    def accuracy(self, predictions, labels):
        return (np.sum(np.argmax(predictions,1) == np.argmax(labels,1)) / predictions.shape[0])
            
    def train(self, x, y):
        feed_dict={self.features: x, self.labels: y, self.is_train: True}
        self.sess.run(self.train_op, feed_dict=feed_dict)
        loss, predict = self.sess.run([self.loss, self.prediction], feed_dict=feed_dict)
        acc = self.accuracy(predict, y)
        return loss, predict, acc
    
    def predict(self, x):
        feed_dict={self.features: x, self.is_train: False}
        predict = self.sess.run(self.prediction, feed_dict=feed_dict)
        return predict
            
    def save(self, path):
        with self.graph.as_default():
            self.saver = tf.train.Saver()
        self.saver.save(self.sess, path) # save_path ex. './Parameter/myVGG_Clean/myVGG_Clean.ckpt'
        print("Successfully save the model !")

    def restore(self, path, keyword=False):
        with self.graph.as_default():
            if keyword:
                var_list = [i for i in tf.global_variables() if keyword in i.name]
                self.saver = tf.train.Saver(var_list=var_list)
            else:
                self.saver = tf.train.Saver()
        self.saver.restore(self.sess, tf.train.latest_checkpoint(path)) # load_path ex. './Parameter/myVGG_Clean/'