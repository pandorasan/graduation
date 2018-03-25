# coding = utf-8
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import math
import matplotlib.pyplot as plt
import pylab
from pylab import *
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x_data = tf.placeholder("float", shape=[None, 784])
y_data = tf.placeholder("float", shape=[None, 10])

MAX_STEP = 5000
NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
train_accu  = []
train_loss = []
sess = tf.InteractiveSession()
def inference(images, hidden1_units, hidden2_units,y_data):
    w0 = tf.Variable(
        tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                            stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
        name='weights')
    b0 = tf.Variable(tf.zeros([hidden1_units]),
                         name='biases')
    w1 = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden2_units],
                            stddev=1.0 / math.sqrt(float(hidden1_units))),
        name='weights')
    b1 = tf.Variable(tf.zeros([hidden2_units]),
                     name='biases')
    w2 = tf.Variable(
        tf.truncated_normal([hidden2_units, NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(hidden2_units))),
        name='weights')
    b2 = tf.Variable(tf.zeros([NUM_CLASSES]),
                     name='biases')
    hidden1 = tf.nn.relu(tf.matmul(images, w0) + b0)
    hidden2 = tf.nn.relu(tf.matmul(hidden1, w1))
    logits = tf.matmul(hidden2, w2) + b2
    hidden2 = tf.nn.relu(tf.matmul(hidden1, w1) + b1 + tf.matmul(logits, tf.transpose(w2)))
    hidden1 = tf.nn.relu(b0+tf.matmul(hidden2, tf.transpose(w1)))
    logits = tf.matmul(hidden2, w2) + b2
    return logits, hidden1, w0, w1, w2, b1, b2


def inference2(y_data, images, hidden1_units, hidden2_units):
    logits = inference(images, hidden1_units, hidden2_units,y_data)
    x = tf.nn.relu(tf.matmul(logits[1], tf.transpose(logits[2])))
    return x


y_pre = inference(x_data, 300, 100, y_data)[0]
x_pre = inference2(y_data, x_data, 300, 100)
loss = tf.losses.mean_squared_error(x_data, x_pre)+\
       tf.losses.softmax_cross_entropy(onehot_labels=y_data, logits=y_pre)
error = tf.losses.mean_squared_error(x_data, x_pre)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(y_data,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())

plt.ion()
N_TEST_IMG = 5
# initialize figure
f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
plt.ion()


for step in range(MAX_STEP):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x_data: batch[0], y_data: batch[1]})
    if step%100 == 0:
        gene_loss = error.eval(feed_dict={x_data:batch[0], y_data: batch[1]})
        loss = gene_loss
        train_accuracy = accuracy.eval(feed_dict={
        x_data:batch[0], y_data: batch[1]})
        accu = train_accuracy
        print ("step %d, training accuracy %g"%(step, train_accuracy))
        print ("loss %g"%(gene_loss))
        train_loss.append(loss)
        train_accu.append(accu)
        # visualization
        view_data = batch[0][:N_TEST_IMG]
        for i in range(N_TEST_IMG):
            # original data (first row) for viewing
            a[0][i].imshow(np.reshape(view_data[i], (28, 28)), cmap='gray')
            a[0][i].set_xticks(());
            a[0][i].set_yticks(())
            a[1][i].clear()
            # generated image
            x_ = tf.reshape(x_pre[i], [28, 28])
            x = x_.eval(session=sess, feed_dict={x_data: batch[0], y_data: batch[1]})
            a[1][i].imshow(x, cmap='gray')
            a[1][i].set_xticks(()); a[1][i].set_yticks(())
        plt.draw(); plt.pause(0.005)


plt.ioff()



