# coding = utf-8
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import pylab
from pylab import *
import lmser_model
# data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x_data = tf.placeholder("float", shape=[None, 784])
y_data = tf.placeholder("float", shape=[None, 10])
MAX_STEP = 5000
NUM_CLASSES = 10
IMAGE_SIZE = 28
train_accu= []
train_loss = []
sess = tf.InteractiveSession()


def back_(x, w, b):
    y = tf.nn.relu(tf.matmul(x, tf.transpose(w))+b)
    return y


def gene(images, hidden1_units, hidden2_units, hidden3_units):
    logits = lmser_model.inference_biminist(images, hidden1_units, hidden2_units, hidden3_units)
    x1 = back_(logits[0], logits[4], logits[7])
    x2 = back_(x1, logits[3], logits[6])
    x3 = back_(x2, logits[2], logits[5])
    x4 = back_(x3,logits[1],0)
    return x4


y_pre = lmser_model.inference_biminist(x_data, 320,160, 80)[0]
x_pre = gene( x_data, 320, 160, 80)
hidden1 = lmser_model.inference3(x_data, 256, 144, 64)[9]
hidden2 = lmser_model.inference3(x_data, 256, 144, 64)[10]


loss = tf.losses.softmax_cross_entropy(onehot_labels=y_data, logits=y_pre)+\
       tf.losses.mean_squared_error(x_data, x_pre)
error = tf.losses.mean_squared_error(x_data, x_pre)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(y_data,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
saver = tf.train.Saver()
plt.ion()
N_TEST_IMG = 5
# initialize figure
f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
plt.ion()
for step in range(MAX_STEP):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x_data: batch[0], y_data: batch[1]})
    err = 0
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
            #generate hidden1
            h1 = tf.reshape(hidden1[i], [16, 16])
            h1_ = h1.eval(session=sess, feed_dict={x_data: batch[0], y_data: batch[1]})
            a[2][i].imshow(h1_, cmap='gray')
            a[2][i].set_xticks(());a[2][i].set_yticks(())
            #generate hidden2
            h2 = tf.reshape(hidden2[i], [12, 12])
            h2_ = h2.eval(session=sess, feed_dict={x_data: batch[0], y_data: batch[1]})
            a[3][i].imshow(h2_, cmap='gray')
            a[3][i].set_xticks(()); a[3][i].set_yticks(())
        plt.draw(); plt.pause(0.01)

np.save('biminist/train_loss', train_loss)
np.save('biminist/train_accu', train_accu)
saver.save(sess, 'biminist/model.ckpt')
plt.ioff()







