'''Trains and evaluates a fully-connected neural net classifier for CIFAR-10'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time
from datetime import datetime
import os.path
import data_helpers
import lmser_inference
import math

# Model parameters as external flags
flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for the training.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 120, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 60, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 30, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('batch_size', 400,
  'Batch size. Must divide dataset sizes without remainder.')
flags.DEFINE_string('train_dir', 'tf_logs',
  'Directory to put the training data.')
flags.DEFINE_float('reg_constant', 0.1, 'Regularization constant.')

#FLAGS._parse_flags()
print('\nParameters:')
for attr, value in sorted(FLAGS.__flags.items()):
  print('{} = {}'.format(attr, value))
print()

IMAGE_PIXELS = 3072
NUM_CLASSES = 10
accusum  = []
losssum = []
#sess = tf.InteractiveSession()
beginTime = time.time()

# Put logs for each run in separate directory
logdir = FLAGS.train_dir + '/' + datetime.now().strftime('%Y%m%d-%H%M%S') + '/'

# Uncommenting these lines removes randomness
# You'll get exactly the same result on each run
# np.random.seed(1)
# tf.set_random_seed(1)

# Load CIFAR-10 data
data_sets = data_helpers.load_data()

# -----------------------------------------------------------------------------
# Prepare the Tensorflow graph
# (We're only defining the graph here, no actual calculations taking place)
# -----------------------------------------------------------------------------

# Define input placeholders
x_data = tf.placeholder('float', shape=[None, IMAGE_PIXELS],
  name='images')
yy_data = tf.placeholder(tf.int64, shape=[None], name='image-labels')
y_data = tf.one_hot(yy_data, 10, axis=1)

# Operation for the classifier's result


hehe = lmser_inference.inference(x_data, IMAGE_PIXELS,
  FLAGS.hidden1, FLAGS.hidden2, FLAGS.hidden3, y_data, NUM_CLASSES, reg_constant=FLAGS.reg_constant)
logits = hehe[0]
x_pre = tf.matmul(hehe[1], tf.transpose(hehe[2]))


loss = lmser_inference.loss(logits, y_data, x_pre, x_data)
# Operation for the training step
train_step = lmser_inference.training(loss, FLAGS.learning_rate)

# Operation calculating the accuracy of our predictions
accuracy = lmser_inference.evaluation(logits, y_data)

# Operation merging summary data for TensorBoard
summary = tf.summary.merge_all()

# Define saver to save model state at checkpoints
saver = tf.train.Saver()

# -----------------------------------------------------------------------------
# Run the TensorFlow graph
# -----------------------------------------------------------------------------

def onehot_(labels):
  batch_size = tf.size(labels)
  labels = tf.expand_dims(labels, 1)
  indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
  concated = tf.concat([indices, labels], 1)
  onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, 10]), 1.0, 0.0)
  return onehot_labels


with tf.Session() as sess:
  # Initialize variables and create summary-writer
  sess.run(tf.global_variables_initializer())
  summary_writer = tf.summary.FileWriter(logdir, sess.graph)

  # Generate input data batches
  zipped_data = zip(data_sets['images_train'], data_sets['labels_train'])
  batches = data_helpers.gen_batch(list(zipped_data), FLAGS.batch_size,
    FLAGS.max_steps)

  for i in range(FLAGS.max_steps):

    # Get next input data batch
    # Get next input data batch
    batch = next(batches)
    images_batch, labels_batch = zip(*batch)
    feed_dict = {
      x_data: images_batch,
      yy_data: labels_batch
    }


    # Periodically print out the model's current accuracy
    if i % 100 == 0:
      train_accuracy = sess.run(accuracy, feed_dict=feed_dict)
      print('Step {:d}, training accuracy {:g}'.format(i, train_accuracy))
      summary_str = sess.run(summary, feed_dict=feed_dict)
      summary_writer.add_summary(summary_str, i)

    # Perform a single training step
    sess.run([train_step, loss], feed_dict=feed_dict)

    # Periodically save checkpoint
    '''
    if (i + 1) % 1000 == 0:
      checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
      saver.save(sess, checkpoint_file, global_step=i)
      print('Saved checkpoint')
      '''

  # After finishing the training, evaluate on the test set
  test_accuracy = sess.run(accuracy, feed_dict={
    x_data: data_sets['images_test'],
    y_data: data_sets['labels_test']})
  print('Test accuracy {:g}'.format(test_accuracy))

endTime = time.time()
print('Total time: {:5.2f}s'.format(endTime - beginTime))
