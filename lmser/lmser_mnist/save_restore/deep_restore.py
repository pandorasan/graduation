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

NUM_LAYERS = 6
HIDDEN_UNITS = [784,300,150,80,40,20,10]
HIDDEN_NAMES = ['hidden1', 'hidden2', 'hidden3', 'hidden4', 'hidden5', 'hidden6']
NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

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

logits = tf.placeholder('float32', [1,10])
hidden.append(logits)
def inference():
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

    for i in range(1,6):
        hidden.append(tf.nn.relu(tf.matmul(hidden[i-1],tf.transpose(names['w%s' % (6-i)])) + names['b%s' % (5-i)]))

    x = tf.nn.relu(tf.matmul(hidden[5], tf.transpose(w0)))
    return x



N_TEST_IMG = 5
# initialize figure
f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
x_pre = inference()

saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())
saver.restore(sess,  "/model.ckpt")

for i in range(5):
    logitshehe = 0.5*np.ones([1, NUM_CLASSES], 'float32')
    logitshehe[0, i] = 1
    print(logitshehe)

    # generated image
    #x_pre = sess.run(x_pre, feed_dict={logits: logitshehe})
    x_ = tf.reshape(x_pre, [28, 28])
    x = x_.eval(session=sess, feed_dict={logits: logitshehe})
    a[0][i].imshow(x, cmap='gray')
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())

for i in range(5,10):
    logitshehe = np.zeros([1, NUM_CLASSES], 'float32')
    logitshehe[0, i] = 1
    # generated image
    x_ = tf.reshape(x_pre, [28, 28])
    x = x_.eval(session=sess, feed_dict={logits: logitshehe})
    a[1][i-5].imshow(x, cmap='gray')
    a[1][i-5].set_xticks(())
    a[1][i-5].set_yticks(())


plt.show()


