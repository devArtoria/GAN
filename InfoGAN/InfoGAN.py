import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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

def Generator(Z, C):
    with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):
        inputs = tf.concat([Z, C], 1)
        hidden = tf.layers.dense(inputs, n_hidden)
        hidden = tf.nn.relu(hidden)
        output = tf.layers.dense(hidden, n_input)
        output = tf.nn.sigmoid(output)

        return output

def Discriminator(X, C):
    with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
        inputs = tf.concat([X, C], 1)
        hidden = tf.layers.dense(inputs, n_hidden)
        hidden = tf.nn.relu(hidden)
        output = tf.layers.dense(hidden, 1)

        return output

def get_noise(batch_size, n_noise):
    return np.random.uniform(-1., 1., size=(batch_size, n_noise))

G = Generator(Z, C)
D_real = Discriminator(X, C)
D_gene = Discriminator(G, C)

loss_D = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)) +
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_gene, labels=tf.zeros_like(D_gene)))
loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_gene, labels=tf.ones_like(D_gene)))

train_params = tf.trainable_variables()
D_var_list = [var for var in train_params if var.name.startswith('D')]
G_var_list = [var for var in train_params if var.name.startswith('G')]

train_D = tf.train.AdamOptimizer(learning_rate).minimize(loss_D, var_list=D_var_list)
train_G = tf.train.AdamOptimizer(learning_rate).minimize(loss_G, var_list=G_var_list)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples / batch_size)
loss_val_D, loss_val_G = 0, 0

for epoch in range(total_epoch):
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        noise = get_noise(batch_size, n_noise)

        _, loss_val_D = sess.run([train_D, loss_D], feed_dict={X: batch_xs, C: batch_ys, Z: noise})
        _, loss_val_G = sess.run([train_G, loss_G], feed_dict={C: batch_ys, Z: noise})

        print('Epoch :  %04d' % epoch, 'D loss: {:.4}'.format(loss_val_D), 'G loss: {:.4}'.format(loss_val_G))

    if epoch == 0 or (epoch+1) % 10 == 0:
        sample_size = 10
        noise = get_noise(sample_size, n_noise)
        samples = sess.run(G, feed_dict={C: mnist.test.labels[:sample_size], Z: noise})

        fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))

        for i in range(sample_size):
            ax[0][i].set_axis_off()
            ax[1][i].set_axis_off()

            ax[1][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
            ax[1][i].imshow(np.reshape(samples[i], (28, 28)))

        plt.savefig('samples/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)

print('Optimization Finished!')
