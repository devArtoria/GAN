import tensorflow as tf

tf.set_random_seed(777)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data", one_hot=True)

total_epoch = 100
batch_size = 100
learning_rate = 0.0002

n_input = 28*28

sample_size = 8
sample_num = 64

z_dim = 128
gf_dim = 64
df_dim = 64

eps = 1e-12

X = tf.placeholder(tf.float32, [])

X = tf.placeholder(tf.float32, [None, n_input])
Z = tf.placeholder(tf.float32, [None, z_dim])



def generator(Z):
    G1 = tf.layers.dense(Z, df_dim, activation=tf.nn.leaky_relu)

    G2 = tf.reshape(G1, [batch_size, 4, 4, gf_dim * 8])
    G2 = tf.layers.batch_normalization(inputs=G2,
                                       momentum=0.9,
                                       epsilon=eps,
                                       scale=True,
                                       training=True)
    G2 = tf.nn.relu(G2)

    G3 = tf.layers.conv2d_transpose(inputs=G2,
                                    filters=gf_dim * 4,
                                    kernel_size=3,
                                    strides=2,
                                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                                    bias_initializer=tf.zeros_initializer(),
                                    padding='SAME')
    G3 = tf.layers.batch_normalization(inputs=G3,
                                       momentum=0.9,
                                       epsilon=eps,
                                       scale=True,
                                       training=True)
    G3 = tf.nn.relu(G3)

    G4 = tf.layers.conv2d_transpose(inputs=G3,
                                    filters=gf_dim * 2,
                                    kernel_size=3,
                                    strides=2,
                                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                                    bias_initializer=tf.zeros_initializer(),
                                    padding='SAME')
    G4 = tf.nn.relu(G4)

    logit = tf.layers.conv2d_transpose(inputs=G4,
                                       filters=1,
                                       kernel_size=3,
                                       strides=2,
                                       kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                                       bias_initializer=tf.zeros_initializer(),
                                       padding='SAME')

    output = tf.nn.tanh(logit)

    return output

