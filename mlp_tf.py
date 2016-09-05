# -*- coding:utf-8 -*-
# Create by steve in 16-9-4 at 下午8:10

from __future__ import print_function

import tensorflow as tf
import numpy as np
#import DataManage as DM


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


if __name__ == '__main__':
    x = np.loadtxt("outX.txt")
    y_t = np.loadtxt("outY.txt")

    print("Without theano!!!!")
    #x, y_t = dm.get_train_set()

    y = np.zeros([x.shape[0], 2], dtype=float)
    for i in range(0, y_t.shape[0] - 1):
        if y_t[i] == 1:
            y[i, 0] = 1
        else:
            y[i, 1] = 1


    print(y.mean())
    # net = buildNetwork(4554, 22, 2, bias=True)
    # ds = ClassificationDataSet(4554, 1, nb_classes=2)
    # t_ds = ClassificationDataSet(4554, 1, nb_classes=2)
    # max_iterations = 50000
    # err_list = np.zeros([max_iterations])

    X = tf.placeholder("float", [None, 4554])
    Y = tf.placeholder("float", [None, 2])

    w_h1 = init_weights([4554, 200])
    w_h2 = init_weights([200, 30])
    w_o = init_weights([30, 2])

    #X = tf.nn.dropout(X,tf.placeholder("float"))

    h1 = tf.nn.sigmoid(tf.matmul(X, w_h1))
    #h1 = tf.nn.dropout(h1,tf.placeholder("float"))

    h2 = tf.nn.sigmoid(tf.matmul(h1, w_h2))
    #h2 = tf.nn.dropout(h2,tf.placeholder("float"))

    py_x = tf.matmul(h2, w_o)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
    train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
    #train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    predict_op = tf.argmax(py_x, 1)

    print("Begin to train model.")
    with tf.Session() as sess:
        tf.initialize_all_variables().run()


        print("init varialbles.")
        for i in range(2000):
            # for start,end in zip(range(0,x.shape[0],100),range(100,x.shape[0],100)):
            #     sess.run(train_op,feed_dict = {X:x[start:end],Y:y[start:end]})
            print("train")
            #7202
            train_N = 22
            batch_size = 200
            for j in range(1, train_N):
                #print("data from : ", (i - 1) * 50, " to ", i * 50)
                sess.run(train_op, feed_dict={X: x[(j - 1) * batch_size+1:j * batch_size,:],
                                              Y: y[(j- 1) * batch_size+1:j * batch_size,:]})

            print ("test",i, np.mean(np.argmax(y[batch_size*train_N::], axis=1) ==
                              sess.run(predict_op, feed_dict={X: x[batch_size*train_N::,:], Y: y[batch_size*train_N::]})))
            print("train",i, np.mean(np.argmax(y[0:batch_size*train_N], axis=1) ==
                             sess.run(predict_op, feed_dict={X: x[0:batch_size*train_N, :], Y: y[0:batch_size*train_N]})))
