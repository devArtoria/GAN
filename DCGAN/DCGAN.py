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


