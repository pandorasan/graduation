# coding = utf-8
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import pylab
from pylab import *
test_loss = []
test_accu = []
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x_data = tf.placeholder("float", shape=[None, 784])
y_data = tf.placeholder("float", shape=[None, 10])

MAX_STEP = 2000
NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

sess = tf.InteractiveSession()
hidden1_units = 256
hidden2_units = 144
hidden3_units = 64


logits = tf.placeholder('float32', [1,10])
w0 = tf.Variable(
    tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                        stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
    name='w0')
b0 = tf.Variable(tf.zeros([hidden1_units]),
                 name='b0')
w1 = tf.Variable(
    tf.truncated_normal([hidden1_units, hidden2_units],
                        stddev=1.0 / math.sqrt(float(hidden1_units))),
    name='w1')
b1 = tf.Variable(tf.zeros([hidden2_units]),
                 name='b1')
w2 = tf.Variable(
    tf.truncated_normal([hidden2_units, hidden3_units],
                        stddev=1.0 / math.sqrt(float(hidden2_units))),
    name='w2')
b2 = tf.Variable(tf.zeros([hidden3_units]),
                 name='b2')

w3 = tf.Variable(
    tf.truncated_normal([hidden3_units, NUM_CLASSES],
                        stddev=1.0 / math.sqrt(float(hidden2_units))),
    name='w3')
b3 = tf.Variable(tf.zeros([NUM_CLASSES]),
                 name='b3')

N_TEST_IMG = 5
# initialize figure
f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))



hidden5 = tf.nn.relu(b5 + tf.matmul(logits, tf.transpose(w5)))
hidden4 = tf.nn.relu(b4 + tf.matmul(hidden5, tf.transpose(w4)))
hidden3 = tf.nn.relu(b3 + tf.matmul(hidden4, tf.transpose(w3)))
hidden2 = tf.nn.relu(b1 + tf.matmul(hidden3, tf.transpose(w2)))
hidden1 = tf.nn.relu(b0 + tf.matmul(hidden2, tf.transpose(w1)))
x_pre = tf.nn.relu(tf.matmul(hidden1, tf.transpose(w0)))

saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())
sess.run(tf.initialize_all_variables())
saver.restore(sess, "deep/model.ckpt")

for i in range(5):
    logitshehe = np.zeros([1, NUM_CLASSES], 'float32')
    logitshehe[0, i] = 2
    # generated image
    #x_pre = sess.run(x_pre, feed_dict={logits: logitshehe})
    x_ = tf.reshape(x_pre, [28, 28])
    x = x_.eval(session=sess, feed_dict={logits: logitshehe})
    a[0][i].imshow(x, cmap='gray')
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())

for i in range(5,10):
    logitshehe = np.zeros([1, NUM_CLASSES], 'float32')
    logitshehe[0, i] = 2
    # generated image
    x_ = tf.reshape(x_pre, [28, 28])
    x = x_.eval(session=sess, feed_dict={logits: logitshehe})
    a[1][i-5].imshow(x, cmap='gray')
    a[1][i-5].set_xticks(())
    a[1][i-5].set_yticks(())


plt.show()


