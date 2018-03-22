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
NUM_LAYERS = 6
HIDDEN_UNITS = [784,100,100,50,25,5,10]
HIDDEN_NAMES = ['hidden', 'hidden2', 'hidden3', 'hidden4', 'hidden5', 'hidden6']
MAX_STEP = 5000
NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
train_accu  = []
train_loss = []
sess = tf.InteractiveSession()


def create_variable(bottomsize, topsize, name):
    with tf.variable_scope(name):
        w = tf.Variable(
            tf.truncated_normal([bottomsize, topsize],
                                stddev=1.0 / math.sqrt(float(bottomsize))),
            name='weights')
        b = tf.Variable(tf.zeros([topsize]),
                                 name='biases')
        return w, b
hidden = []


def inference(image):
    w0, b0 = create_variable(HIDDEN_UNITS[0], HIDDEN_UNITS[1], HIDDEN_NAMES[0])
    w1, b1 = create_variable(HIDDEN_UNITS[1], HIDDEN_UNITS[2], HIDDEN_NAMES[1])
    w2, b2 = create_variable(HIDDEN_UNITS[2], HIDDEN_UNITS[3], HIDDEN_NAMES[2])
    w3, b3 = create_variable(HIDDEN_UNITS[3], HIDDEN_UNITS[4], HIDDEN_NAMES[3])
    w4, b4 = create_variable(HIDDEN_UNITS[4], HIDDEN_UNITS[5], HIDDEN_NAMES[4])
    w5, b5 = create_variable(HIDDEN_UNITS[5], NUM_CLASSES, HIDDEN_NAMES[5])
    names = locals()

    '''
        for i in range(5):
        names['x%s' % i] = create_variable(HIDDEN_UNITS[0], HIDDEN_UNITS[1], HIDDEN_NAMES[0])
    '''
    hidden.append(image)
    for jj in range(1,NUM_LAYERS+1):
        hid_num = HIDDEN_UNITS[jj]
        hidden.append(np.zeros((1,hid_num),dtype='float32'))

    for i in range(1,NUM_LAYERS):
        hidden[i] = tf.nn.relu(tf.matmul(hidden[i-1], names['w%s' % (i-1)]) +\
                               tf.matmul(hidden[i+1],tf.transpose(names['w%s' % i])) + names['b%s' % (i-1)])
    hidden[NUM_LAYERS] = tf.matmul(hidden[NUM_LAYERS-1], w5) + b5

    for i in range(NUM_LAYERS-1,0):
        hidden[i] = tf.nn.relu(tf.matmul(hidden[i-1], names['w%s' % (i-1)]) +\
                               tf.matmul(hidden[i+1],tf.transpose(names['w%s' % i])) + names['b%s' % (i-1)])
    hidden[NUM_LAYERS] = tf.matmul(hidden[NUM_LAYERS-1], w5) + b5

    return hidden[NUM_LAYERS], hidden[1], w0


def inference2(image):
    logits = inference(image)
    x = tf.nn.relu(tf.matmul(logits[1], tf.transpose(logits[2])))
    return x


y_pre = inference(x_data)[0]
x_pre = inference2(x_data)
loss = tf.losses.mean_squared_error(x_data, x_pre)
#tf.losses.softmax_cross_entropy(onehot_labels=y_data, logits=y_pre)

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
        print "step %d, training accuracy %g"%(step, train_accuracy)
        print "loss %g"%(gene_loss)
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
np.save('tflmser_train_loss', train_loss)
np.save('tflmser_train_accu', train_accu)


