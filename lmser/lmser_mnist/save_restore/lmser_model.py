# coding = utf-8
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import math
import matplotlib.pyplot as plt
import pylab
from pylab import *
# 2 layers
IMAGE_PIXELS = 784
NUM_CLASSES = 10


def inference3_best(images, hidden1_units, hidden2_units,hidden3_units,y_data):
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
                            stddev=1.0 / math.sqrt(float(hidden3_units))),
        name='w2')
    b2 = tf.Variable(tf.zeros([hidden3_units]),
                     name='b2')
    w3 = tf.Variable(
        tf.truncated_normal([hidden3_units, NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(hidden2_units))),
        name='w3')
    b3 = tf.Variable(tf.zeros([NUM_CLASSES]),
                     name='b3')
    hidden1 = tf.nn.relu(tf.matmul(images, w0) + b0)
    hidden2 = tf.nn.relu(tf.matmul(hidden1, w1) + b1)
    hidden3 = tf.nn.relu(tf.matmul(hidden2, w2) + b2 + tf.matmul(y_data, tf.transpose(w3)))
    hidden2 = tf.nn.relu(tf.matmul(hidden1, w1) + b1 + tf.matmul(hidden3, tf.transpose(w2)))
    hidden1 = tf.nn.relu(b0 + tf.matmul(hidden2, tf.transpose(w1))+tf.matmul(images, w0))
    logits = tf.matmul(hidden3, w3) + b3
    return logits, hidden1, w0


def generate_best(images, hidden1_units, hidden2_units,hidden3_units,y_data):
    logits = inference3_best(images, hidden1_units, hidden2_units, hidden3_units,y_data)
    x = tf.nn.relu(tf.matmul(logits[1], tf.transpose(logits[2])))
    return x




def inference2(images, hidden1_units, hidden2_units,hidden3_units):
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
                            stddev=1.0 / math.sqrt(float(hidden3_units))),
        name='w2')
    b2 = tf.Variable(tf.zeros([hidden3_units]),
                     name='b2')
    hidden1 = tf.nn.relu(tf.matmul(images, w0) + b0)
    hidden2 = tf.nn.relu(tf.matmul(hidden1, w1))
    logits = tf.matmul(hidden2, w2) + b2
    hidden2 = tf.nn.relu(tf.matmul(hidden1, w1) + b1 + tf.matmul(logits, tf.transpose(w2)))
    hidden1 = tf.nn.relu(b0+tf.matmul(hidden2, tf.transpose(w1)))
    logits = tf.matmul(hidden2, w2) + b2
    return logits, hidden1, w0


def inference3(images, hidden1_units, hidden2_units,hidden3_units):
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
                            stddev=1.0 / math.sqrt(float(hidden3_units))),
        name='w2')
    b2 = tf.Variable(tf.zeros([hidden3_units]),
                     name='b2')
    w3 = tf.Variable(
        tf.truncated_normal([hidden3_units, NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(hidden2_units))),
        name='w3')
    b3 = tf.Variable(tf.zeros([NUM_CLASSES]),
                     name='b3')
    hidden1 = tf.nn.relu(tf.matmul(images, w0) + b0)
    hidden2 = tf.nn.relu(tf.matmul(hidden1, w1) + b1)
    hidden3 = tf.nn.relu(tf.matmul(hidden2, w2) + b2)
    logits = tf.matmul(hidden3, w3) + b3
    hidden3 = tf.nn.relu(tf.matmul(hidden2, w2) + b2 + tf.matmul(logits, tf.transpose(w3)))
    hidden2 = tf.nn.relu(tf.matmul(hidden1, w1) + b1 + tf.matmul(hidden3, tf.transpose(w2)))
    hidden1 = tf.nn.relu(b0+tf.matmul(hidden2, tf.transpose(w1))+tf.matmul(images, w0))
    logits = tf.matmul(hidden3, w3) + b3
    return logits,hidden1, w0, b0, w1, b1, w2, b2, w3, b3, hidden1, hidden2

def inference3_biminist(images, hidden1_units, hidden2_units, hidden3_units):
    # Hidden 1
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
                            stddev=1.0 / math.sqrt(float(hidden3_units))),
        name='w2')
    b2 = tf.Variable(tf.zeros([hidden3_units]),
                     name='b2')
    w3 = tf.Variable(
        tf.truncated_normal([hidden3_units, NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(hidden2_units))),
        name='w3')
    b3 = tf.Variable(tf.zeros([NUM_CLASSES]),
                     name='b3')
    hidden1 = tf.nn.relu(tf.matmul(images, w0) + b0)
    hidden2 = tf.nn.relu(tf.matmul(hidden1, w1) + b1)
    hidden3 = tf.nn.relu(tf.matmul(hidden2, w2) + b2)
    logits = tf.matmul(hidden3, w3) + b3
    return logits, w0, w1, w2, w3, b0, b1, b2, b3,hidden1, hidden2



def inference2_biminist(images, hidden1_units, hidden2_units):
    # Hidden 1
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
        tf.truncated_normal([hidden2_units, NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(NUM_CLASSES))),
        name='w2')
    b2 = tf.Variable(tf.zeros([NUM_CLASSES]))
    hidden1 = tf.nn.relu(tf.matmul(images, w0) + b0)
    hidden2 = tf.nn.relu(tf.matmul(hidden1, w1) + b1)
    logits = tf.matmul(hidden2, w2) + b2
    return logits, w0, w1, w2, b0, b1, b2, hidden1, hidden2

def generate_2(images, hidden1_units, hidden2_units):
    logits = inference2(images, hidden1_units, hidden2_units)
    x = tf.nn.relu(tf.matmul(logits[1], tf.transpose(logits[2])))
    return x

def generate_3(images, hidden1_units, hidden2_units, hidden3_units):
    logits = inference3(images, hidden1_units, hidden2_units, hidden3_units)
    x = tf.nn.relu(tf.matmul(logits[1], tf.transpose(logits[2])))
    return x


def generate2_biminist(images, hidden1_units, hidden2_units):
    def back_(x, w, b):
        y = tf.nn.relu(tf.matmul(x, tf.transpose(w)) + b)
        return y
    logits = inference2_biminist(images, hidden1_units, hidden2_units)
    x1 = back_(logits[0], logits[3], logits[5])
    x2 = back_(x1, logits[2], logits[4])
    x3 = back_(x2, logits[1],0)
    return x3



def generate3_biminist(images, hidden1_units, hidden2_units,hidden3_units):
    def back_(x, w, b):
        y = tf.nn.relu(tf.matmul(x, tf.transpose(w)) + b)
        return y
    logits = inference3_biminist(images, hidden1_units, hidden2_units, hidden3_units)
    x1 = back_(logits[0], logits[4], logits[7])
    x2 = back_(x1, logits[3], logits[6])
    x3 = back_(x2, logits[2], logits[5])
    x4 = back_(x3,logits[1],0)
    return x4


def load_checkpoint(sess, var_list, checkpoint_path):
    saver = tf.train.Saver(var_list)
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    tf.logging.info('loading model %s', ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)


def save_model(sess, var_list, model_save_path, global_step):
    saver = tf.train.Saver(var_list)
    check_point = os.path.join(model_save_path,'vector')
    saver.save(sess, 'my_lmser_model',global_step=1000)


def load_model(sess, trained_model):
    saver = tf.train.import_meta_graph(trained_model)
    saver.restore(sess,tf.train.latest_checkpoint('./'))


