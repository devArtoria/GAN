import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data", one_hot=True, reshape=[])

batch_size = 100
learning_rate = 0.0002
epoch = 20

n_height = 64
n_width = 64
n_noise = 100

X = tf.placeholder(tf.float32, [None, n_height, n_width, 1])
Z = tf.placeholder(tf.float32, [None, 1, 1, n_noise])
training = tf.placeholder(tf.bool)

def Generator(Z, reuse=False, training=True):
    with tf.variable_scope("Generator", reuse=tf.AUTO_REUSE):
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

def Discriminator(X, reuse=False, training=True):
    with tf.variable_scope("Discriminator", reuse=tf.AUTO_REUSE):
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

G = Generator(Z, training=training)

D_gene, D_gene_logits = Discriminator(G, True, training)
D_real, D_real_logits = Discriminator(X, training=training)

loss_D = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([batch_size, 1, 1, 1])) +
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_gene_logits, labels=tf.zeros([batch_size, 1, 1, 1])))
loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_gene_logits, labels=tf.ones([batch_size, 1, 1, 1])))

train_params = tf.trainable_variables()
D_var_list = [var for var in train_params if var.name.startswith('D')]
G_var_list = [var for var in train_params if var.name.startswith('G')]

train_D = tf.train.AdamOptimizer(learning_rate).minimize(loss_D, var_list=D_var_list)
train_G = tf.train.AdamOptimizer(learning_rate).minimize(loss_G, var_list=G_var_list)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

train_set = tf.image.resize_images(mnist.train.images, [64, 64]).eval()
train_set = (train_set - 0.5) / 0.5

total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(epoch):
    for i in range(total_batch):
        batch_xs = train_set[i*batch_size:(i+1)*batch_size]
        noise = np.random.normal(0, 1, (batch_size, 1, 1, 100))

        _, loss_val_D = sess.run([train_D, loss_D], feed_dict={X: batch_xs, Z: noise, training: True})
        _, loss_val_G = sess.run([train_G, loss_G], feed_dict={X: batch_xs, Z: noise, training: True})

        print('Epoch :  %04d' % epoch, 'D loss: {:.4}'.format(loss_val_D), 'G loss: {:.4}'.format(loss_val_G))

