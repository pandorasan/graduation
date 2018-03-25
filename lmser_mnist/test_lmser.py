# coding = utf-8
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import math
import matplotlib.pyplot as plt
import pylab
from pylab import *
import lmser_model
test_loss = []
test_accu = []

sess = tf.InteractiveSession()
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x_data = tf.placeholder("float", shape=[None, 784])
y_data = tf.placeholder("float", shape=[None, 10])


y_pre = lmser_model.inference3(x_data, 256, 128, 64)[0]
x_pre = lmser_model.generate_3(x_data, 256, 128, 64)
hidden2 = lmser_model.inference3(x_data, 256, 128, 64)[10]
loss = tf.losses.softmax_cross_entropy(onehot_labels=y_data, logits=y_pre)+\
       tf.losses.mean_squared_error(x_data, x_pre)
error = tf.losses.mean_squared_error(x_data, x_pre)
correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(y_data,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
saver = tf.train.Saver()

plt.ion()
N_TEST_IMG = 5
# initialize figure
f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 3))
plt.ion()


sess.run(tf.global_variables_initializer())
saver.restore(sess, "model_3_256/model.ckpt")
for step in range(5000):
    batch = mnist.test.next_batch(50)
    if step%100 == 0:
        gene_loss = error.eval(feed_dict={x_data: batch[0], y_data: batch[1]})
        loss = gene_loss
        test_accuracy = accuracy.eval(feed_dict={
            x_data: batch[0], y_data: batch[1]})
        accu = test_accuracy
        print ("step %d, test accuracy %g" % (step, test_accuracy))
        print ("loss %g" % (gene_loss))
        test_loss.append(loss)
        test_accu.append(accu)
        # visualization
        view_data = batch[0][:N_TEST_IMG]
        for i in range(N_TEST_IMG):
            # original data (first row) for viewing
            a[0][i].imshow(np.reshape(view_data[i], (28, 28)), cmap='gray')
            a[0][i].set_xticks(())
            a[0][i].set_yticks(())
            a[1][i].clear()
            # generated image
            x_ = tf.reshape(x_pre[i], [28, 28])
            x = x_.eval(session=sess, feed_dict={x_data: batch[0], y_data: batch[1]})
            a[1][i].imshow(x, cmap='gray')
            a[1][i].set_xticks(())
            a[1][i].set_yticks(())
            h2 = tf.reshape(hidden2[i], [16, 16])
            h2_ = h2.eval(session=sess, feed_dict={x_data: batch[0], y_data: batch[1]})
            a[2][i].imshow(h2_, cmap='gray')
            a[2][i].set_xticks(())
            a[2][i].set_yticks(())


        plt.draw()
        plt.pause(0.005)
plt.ioff()



