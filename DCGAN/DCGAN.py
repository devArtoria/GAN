import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data", one_hot=True)

batch_size = 100
learning_rate = 0.0002
epoch = 20

n_height = 64
n_width = 64
n_noise = 100

X = tf.placeholder(tf.float32, [None, n_height, n_width, 1])
Z = tf.placeholder(tf.float32, [None, 1, 1, n_noise])
training = tf.placeholder(tf.bool)

def Generator(Z, reuse=None, training=True):
    with tf.variable_scope("Generator", reuse=reuse):
        G1 = tf.layers.conv2d_transpose(Z, 1024, [4, 4], 1, padding='valid')
        G1 = tf.layers.batch_normalization(G1, training=training)
        G1 = tf.nn.leaky_relu(G1)

        G2 = tf.layers.conv2d_transpose(G1, 512, [4, 4], 2, padding='same')
        G2 = tf.layers.batch_normalization(G2, training=training)
        G2 = tf.nn.leaky_relu(G2)

        G3 = tf.layers.conv2d_transpose(G2, 256, [4, 4], 2, padding='same')
        G3 = tf.layers.batch_normalization(G3, training=training)
        G3 = tf.nn.leaky_relu(G3)

        G4 = tf.layers.conv2d_transpose(G3, 128, [4, 4], 2, padding='same')
        G4 = tf.layers.batch_normalization(G4, training=training)
        G4 = tf.nn.leaky_relu(G4)

        G5 = tf.layers.conv2d_transpose(G4, 1, [4, 4], 2, padding='same')
        output = tf.nn.tanh(G5)

        return output

def Discriminator(X, reuse=None, training=True):
    with tf.variable_scope("Discriminator", reuse=reuse):
        D1 = tf.layers.conv2d(X, 128, [4, 4], 2, padding='same')
        D1 = tf.nn.leaky_relu(D1)

        D2 = tf.layers.conv2d(D1, 256, [4, 4], 2, padding='same')
        D2 = tf.layers.batch_normalization(D2, training=training)
        D2 = tf.nn.relu(D2)

        D3 = tf.layers.conv2d(D2, 512, [4, 4], 2, padding='same')
        D3 = tf.layers.batch_normalization(D3, training=training)
        D3 = tf.nn.leaky_relu(D3)

        D4 = tf.layers.conv2d(D3, 1024, [4, 4], 2, padding='same')
        D4 = tf.layers.batch_normalization(D4, training=training)
        D4 = tf.nn.leaky_relu(D4)

        D5 = tf.layers.conv2d(D4, 1, [4, 4], 1, padding='valid')
        output = tf.nn.tanh(D5)

        return output, D5

