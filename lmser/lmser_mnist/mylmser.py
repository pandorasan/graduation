# utf -8
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


BATCH_SIZE = 50
LR = 0.001              # learning rate

mnist = input_data.read_data_sets('./mnist', one_hot=True)  # they has been normalized to range (0,1)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]
xdim = 784


def sigmoid(x):
    y = 1./(1+np.exp(-x))
    return y


def de_sigmoid(x):
    y = np.multiply(x, (1-x))
    return y


epsilon = 0.1
K = 3
N = [2, 3, 2]

Y = [np.zeros((N[0], BATCH_SIZE)), np.zeros((N[1], BATCH_SIZE)), np.zeros((N[2], BATCH_SIZE))]
W = [np.zeros((N[0], xdim)), np.zeros((N[1], N[0])), np.zeros((N[2], N[1]))]
Z = [np.zeros((N[0], BATCH_SIZE)), np.zeros((N[1], BATCH_SIZE)), np.zeros((N[2], BATCH_SIZE))]
U = [np.zeros(xdim,BATCH_SIZE), np.zeros((N[0], BATCH_SIZE)), np.zeros((N[1], BATCH_SIZE))]
E = [np.zeros((N[0], BATCH_SIZE)), np.zeros((N[1], BATCH_SIZE)), np.zeros((N[2], BATCH_SIZE))]

w1 = [np.zeros((3, N[0])), np.zeros((2, 3))]
w2 = [np.zeros((3, N[0])), np.zeros((2, 3))]

for step in range(100):

    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    input = b_x.T
    print(W[0])
    Y[0] = W[0].dot(input)
    Z[0] = sigmoid(Y[0])
    Y[1] = W[1].dot(Z[0])
    Z[1] = sigmoid(Y[1])
    Y[2] = W[2] .dot(Z[1])
    Z[2] = sigmoid(Y[2])

    U[0] = W[0].T.dot(Z[0])
    U[1] = W[1].T.dot(Z[1])
    E0 = input - U0
    E[0] = W[0].dot(E0)
    E[1] = W[1].dot(E[0])

    w10 = np.multiply(np.tile((input-U0).T, (N[0], 1)), np.tile(np.reshape(Z[0],(100,1)), (1, xdim)))
    w20 = np.multiply(np.tile(np.reshape(np.multiply(de_sigmoid((Y[0]+U[0])), E[0]),(100,1)), (1, N[0])), np.tile(input.T, (N[1], 1)))
    dw00 = w10 + w20
    print(np.shape(dw00))
    dw0 = np.zeros((2,784))

    dw0[0,:] = np.mat(np.mean(dw00[0:50, :], 0))
    dw0[1,:] = np.mat(np.mean(dw00[50:99, :],0))
    W[0] = W[0] + epsilon*np.mat(dw0)
    Y[0] = W[0].dot(input)
    Z[0] = sigmoid(Y[0])
    U0 = W[0].T .dot(Z[0])
    E[0] = W[0].dot(E0)

    for k in range(1, 2):
        yue = np.multiply(de_sigmoid((Y[k-1]+U[k-1])), E[k-1]).T
        print(np.shape(yue))
        yue = np.tile(yue, (N[k+1], 1))
        print(np.shape(yue))
        zk = np.tile(np.reshape(Z[k], (50*N[k+1], 1)), (1, N[k]))
        w1 = np.multiply(yue, zk)
        print(np.shape(w1))

        yue = np.multiply(de_sigmoid((Y[k] + U[k])), E[k]).T
        print(np.shape(yue))
        yue = np.tile(np.reshape(yue, (50*N[k+1],1)), (1,N[k]))
        print(np.shape(yue))
        zk = np.tile(Z[k-1].T, (N[k+1], 1))
        w2 = np.multiply(yue, zk)
        dw0 = w1 + w2
        print(np.shape(dw0))
        dw = np.zeros((N[k+1],N[k]))
        print(dw)
        for jj in range(0, N[k+1]):
            x = jj*50
            y = (jj+1)*50
            dww = np.mat(np.mean(dw0[x:y, :], 0))
            dw[jj,:] = dww






        print(np.shape(dww))
        print(np.shape(dw))
        W[k] = W[k] + epsilon*np.mat(dw)
        Y[k] = W[k].dot(Z[k-1])
        Z[k] = sigmoid(Y[k])
        U[k-1] = W[k].T.dot(Z[k])
        E[k] = W[k].dot(E[k-1])

    U0 = W[0].T.dot(Z[0])
    err = sum(sum(np.square(input-U0)))




































