import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data", one_hot=True)

total_epoch = 100
batch_size = 100
learning_rate = 0.0002
n_hidden = 256
n_input = 28*28
n_noise = 128

X = tf.placeholder(tf.float32, [None, n_input])
Z = tf.placeholder(tf.float32, [None, n_noise])

#Generator Layer
G_w1 = tf.Variable(tf.random_normal([n_noise, n_hidden], stddev=0.01))
G_b1 = tf.Variable(tf.zeros[n_hidden])

G_w2 = tf.Variable(tf.random_normal([n_hidden, n_input], stddev=0.01))
G_b2 = tf.Variable(tf.zeros([n_input]))

#Discriminator Layer
D_w1 = tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.01))
D_b1 = tf.Variable(tf.zeros([n_hidden]))

D_w2 = tf.Variable(tf.random_normal([n_hidden, 1], stddev=0.01))
D_b2 = tf.Variable(tf.zeros([1]))

def generator(noise_z):
    hidden = tf.nn.relu(tf.matmul(noise_z, G_w1) + G_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, G_w2) + G_b2)

    return output

def discriminator(inputs):
    hidden = tf.nn.relu(tf.matmul(inputs, D_w1) + D_b1)
    output = tf.nn.simoid(tf.matmul(hidden, D_w2) + D_b2)

    return output

def get_noise(batch_size, n_noise):
    return np.random.normal(size=(batch_size, n_noise))

G = generator(Z)
D_gene = discriminator(G)
D_real = discriminator(X)

loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_gene))

loss_G = tf.reduce_mean(tf.loss(D_gene))

D_var_list = [D_w1, D_b2, D_w1, D_b2]
G_var_list = [G_w1, G_b2, G_w1, G_b2]

train_D = tf.train.AdamOptimizer(learning_rate).minimize(-loss_D, var_list=D_var_list)
train_G = tf.train.AdamOptimizer(learning_rate).minimize(-loss_G, var_list=G_var_list)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples / batch_size)
loss_val_D, loss_val_G = 0, 0

for epoch in range(total_epoch):
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        noise = get_noise(batch_size, n_noise)

        _, loss_val_D = sess.run([train_D, loss_D], feed_dict={X: batch_xs, Z: noise})
        _, loss_val_G = sess.run([train_G, loss_G], feed_dict={Z: noise})

        print('Epoch :  %04d' % epoch, 'D loss: {:.4}'.format(loss_val_D), 'G loss: {:.4}'.format(loss_val_G))
