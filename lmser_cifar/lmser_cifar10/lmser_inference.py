'''Builds a 2-layer fully-connected neural network'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


def inference(images, image_pixels, hidden1_units, hidden2_units ,hidden3_units, y_data,  classes, reg_constant=0):
  '''Build the model up to where it may be used for inference.

  Args:
      images: Images placeholder (input data).
      image_pixels: Number of pixels per image.
      hidden_units: Size of the first (hidden) layer.
      classes: Number of possible image classes/labels.
      reg_constant: Regularization constant (default 0).

  Returns:
      logits: Output tensor containing the computed logits.
  ''' 
  w0 = tf.get_variable(
      name='w0',
      shape=[image_pixels, hidden1_units],
      initializer=tf.truncated_normal_initializer(
        stddev=1.0 / np.sqrt(float(image_pixels))),
      regularizer=tf.contrib.layers.l2_regularizer(reg_constant))
  b0 = tf.Variable(tf.zeros([hidden1_units]), name='b0')
  w1 = tf.get_variable('w1', [hidden1_units, hidden2_units],
      initializer=tf.truncated_normal_initializer(
        stddev=1.0 / np.sqrt(float(hidden2_units))),
      regularizer=tf.contrib.layers.l2_regularizer(reg_constant))
    
  b1 = tf.Variable(tf.zeros([hidden2_units]), name='b1')
  w2 = tf.get_variable('w2', [hidden2_units, hidden3_units],
      initializer=tf.truncated_normal_initializer(
        stddev=1.0 / np.sqrt(float(hidden3_units))),
      regularizer=tf.contrib.layers.l2_regularizer(reg_constant))
  b2 = tf.Variable(tf.zeros([hidden3_units]), name='b2')
    
    
  w3 = tf.get_variable('w3', [hidden3_units, classes],
                         initializer=tf.truncated_normal_initializer(
                             stddev=1.0 / np.sqrt(float(classes))),
                         regularizer=tf.contrib.layers.l2_regularizer(reg_constant))
  b3 = tf.Variable(tf.zeros([classes]), name='b3')
    
  # Define the layer's output
  hidden1 = tf.nn.relu(tf.matmul(images, w0) + b0)
  hidden2 = tf.nn.relu(tf.matmul(hidden1, w1) + b1)
  hidden3 = tf.nn.relu(tf.matmul(hidden2, w2) + b2 + tf.matmul(y_data, tf.transpose(w3)))
  hidden2 = tf.nn.relu(tf.matmul(hidden1, w1) + b1 + tf.matmul(hidden3, tf.transpose(w2)))
  hidden1 = tf.nn.relu(tf.matmul(images, w0) + b0+tf.matmul(hidden2, tf.transpose(w1)))
  logits = tf.matmul(hidden3, w3) + b3

  return logits, hidden1, w0


def loss(y_pre, y_data, x_pre, x_data):
  '''Calculates the loss from logits and labels.

  Args:
    logits: Logits tensor, float - [batch size, number of classes].
    labels: Labels tensor, int64 - [batch size].

  Returns:
    loss: Loss tensor of type float.
  '''

  with tf.name_scope('Loss'):
    # Operation to determine the cross entropy between y_pre and y_data



    # Operation for the loss function
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y_data, logits=y_pre) + tf.add_n(tf.get_collection(
      tf.GraphKeys.REGULARIZATION_LOSSES))
    #+tf.losses.mean_squared_error(x_data, x_pre)

    # Add a scalar summary for the loss
    tf.summary.scalar('loss', loss)

  return loss


def training(loss, learning_rate):
  '''Sets up the training operation.

  Creates an optimizer and applies the gradients to all trainable variables.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

  Returns:
    train_step: The op for training.
  '''

  # Create a variable to track the global step
  global_step = tf.Variable(0, name='global_step', trainable=False)

  # Create a gradient descent optimizer
  # (which also increments the global step counter)
  train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
    loss, global_step=global_step)

  return train_step


def evaluation(y_pre, y_data):
  '''Evaluates the quality of the logits at predicting the label.

  Args:
    logits: Logits tensor, float - [batch size, number of classes].
    labels: Labels tensor, int64 - [batch size].

  Returns:
    accuracy: the percentage of images where the class was correctly predicted.
  '''

  correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y_data, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

  # Summary operation for the accuracy
  tf.summary.scalar('train_accuracy', accuracy)

  return accuracy
