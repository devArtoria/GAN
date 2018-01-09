import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data", one_hot=True)

total_epoch = 100
batch_size = 100
learning_rate = 0.0002
n_hidden = 256
n_input = 28*28
n_noise = 128
n_class = 10

X = tf.placeholder(tf.float32, [None, n_input])
Z = tf.placeholder(tf.float32, [None, n_noise])
C = tf.placeholder(tf.float32, [None, n_class])

def Generator(Z, labels):
    with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):
        inputs = tf.concat([Z, labels], 1)
        hidden = tf.layers.dense(inputs, n_hidden)
        hidden = tf.nn.relu(hidden)
        output = tf.layers.dense(hidden, n_input)
        output = tf.nn.sigmoid(output)

        return output

def Discriminator(X, labels):
    with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
        inputs = tf.concat([X, labels], 1)
        hidden = tf.layers.dense(inputs, n_hidden)
        hidden = tf.nn.relu(hidden)
        output = tf.layers.dense(hidden, 1)

        return output

def get_noise(batch_size, n_noise):
    return np.random.uniform(-1., 1., size=(batch_size, n_noise))