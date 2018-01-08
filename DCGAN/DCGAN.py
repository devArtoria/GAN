import tensorflow as tf

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

