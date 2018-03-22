# coding = utf-8
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import pylab
from pylab import *

NUM_CLASSES = 10
NUM_LAYERS = 6
HIDDEN_UNITS = [784,144,64,30,40,20,10]
HIDDEN_NAMES = ['hidden', 'hidden2', 'hidden3', 'hidden4', 'hidden5', 'hidden6']
N_TEST_IMG = 5
sess = tf.InteractiveSession()


logits = tf.placeholder('float32', [1,10])


def create_variable(bottomsize, topsize, name):
    with tf.variable_scope(name):
        w = tf.Variable(
            tf.truncated_normal([bottomsize, topsize],
                                stddev=1.0 / math.sqrt(float(bottomsize))),
            name='weights')

        b = tf.Variable(tf.zeros([topsize]),
                                 name='biases')
        return w, b




w0, b0 = create_variable(HIDDEN_UNITS[0], HIDDEN_UNITS[1], HIDDEN_NAMES[0])
w1, b1 = create_variable(HIDDEN_UNITS[1], HIDDEN_UNITS[2], HIDDEN_NAMES[1])
w2, b2 = create_variable(HIDDEN_UNITS[2], HIDDEN_UNITS[3], HIDDEN_NAMES[2])
w3, b3 = create_variable(HIDDEN_UNITS[3], HIDDEN_UNITS[4], HIDDEN_NAMES[3])
w4, b4 = create_variable(HIDDEN_UNITS[4], HIDDEN_UNITS[5], HIDDEN_NAMES[4])
w5, b5 = create_variable(HIDDEN_UNITS[5], NUM_CLASSES, HIDDEN_NAMES[5])

hidden5 = tf.nn.relu(b4 + tf.matmul(logits, tf.transpose(w5)))
hidden4 = tf.nn.relu(b3 + tf.matmul(hidden5, tf.transpose(w4)))
hidden3 = tf.nn.relu(b2 + tf.matmul(hidden4, tf.transpose(w3)))
hidden2 = tf.nn.relu(b1 + tf.matmul(hidden3, tf.transpose(w2)))
hidden1 = tf.nn.relu(b0 + tf.matmul(hidden2, tf.transpose(w1)))
x_pre = tf.nn.relu(tf.matmul(hidden1, tf.transpose(w0)))
f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))

saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())
sess.run(tf.initialize_all_variables())
saver.restore(sess, "deep_thin/model.ckpt")

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


