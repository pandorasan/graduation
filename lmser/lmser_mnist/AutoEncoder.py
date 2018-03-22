"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
tensorflow: 1.1.0
matplotlib
numpy
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

tf.set_random_seed(1)

# Hyper Parameters
BATCH_SIZE = 50
LR = 0.0001        # learning rate
N_TEST_IMG = 5

# Mnist digits
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)     # use not one-hotted target data
test_x = mnist.test.images[:200]
test_y = mnist.test.labels[:200]

# plot one example
print(mnist.train.images.shape)     # (55000, 28 * 28)
print(mnist.train.labels.shape)     # (55000, 10)

# tf placeholder
tf_x = tf.placeholder(tf.float32, [None, 28*28])    # value in the range of (0, 1)

# encoder
en0 = tf.layers.dense(tf_x, 256, tf.nn.tanh)
en1 = tf.layers.dense(en0, 128, tf.nn.tanh)
en2 = tf.layers.dense(en1, 64, tf.nn.tanh)
encoded = tf.layers.dense(en2, 10)

# decoder
de0 = tf.layers.dense(encoded, 64, tf.nn.tanh)
de1 = tf.layers.dense(de0, 128, tf.nn.tanh)
de2 = tf.layers.dense(de1, 256, tf.nn.tanh)
decoded = tf.layers.dense(de2, 28*28, tf.nn.sigmoid)

loss = tf.losses.mean_squared_error(labels=tf_x, predictions=decoded)
train = tf.train.AdamOptimizer(LR).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
autoloss = []
# initialize figure
f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
plt.ion()   # continuously plot

# original data (first row) for viewing



for step in range(10000):
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    _, encoded_, decoded_, loss_ = sess.run([train, encoded, decoded, loss], {tf_x: b_x})

    if step % 100 == 0:     # plotting
        print('train loss: %.4f' % loss_)
        # plotting decoded image (second row)
        decoded_data = sess.run(decoded, {tf_x: b_x})
        loss1 = loss_
        autoloss.append(loss1)
        for i in range(N_TEST_IMG):
            view_data = b_x
            decoded_data = sess.run(decoded, {tf_x: view_data})
            a[0][i].imshow(np.reshape(view_data[i], (28, 28)), cmap='gray')
            a[0][i].set_xticks(());
            a[0][i].set_yticks(())
            a[1][i].clear()
            a[1][i].imshow(np.reshape(decoded_data[i], (28, 28)), cmap='gray')
            a[1][i].set_xticks(()); a[1][i].set_yticks(())
        plt.draw(); plt.pause(0.01)
plt.ioff()



# visualize in 3D plot