# -*- coding:utf-8 -*-
# Create by steve in 16-9-4 at 下午8:10

from __future__ import print_function

import tensorflow as tf
import numpy as np
#import DataManage as DM

import sys

import timeit
import time


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


if __name__ == '__main__':
    x = np.loadtxt("outX.txt")
    y_t = np.loadtxt("outY.txt")

    x2 = np.loadtxt("sumX.txt")
    y_t2 = np.loadtxt("sunY.txt")

    print("Without theano!!!!")
    #x, y_t = dm.get_train_set()

    y = np.zeros([x.shape[0], 2], dtype=float)
    for i in range(0, y_t.shape[0] - 1):
        if y_t[i] == 1:
            y[i, 0] = 1
        else:
            y[i, 1] = 1

    y2 = np.zeros([x2.shape[0], 2], dtype=float)
    for i in range(0, y_t2.shape[0] - 1):
        if y_t2[i] == 1:
            y2[i, 0] = 1
        else:
            y2[i, 1] = 1



    print(x.mean())
    print(y_t.mean())
    # net = buildNetwork(4554, 22, 2, bias=True)
    # ds = ClassificationDataSet(4554, 1, nb_classes=2)
    # t_ds = ClassificationDataSet(4554, 1, nb_classes=2)
    # max_iterations = 50000
    # err_list = np.zeros([max_iterations])

    X = tf.placeholder("float", [None, 4554])
    Y = tf.placeholder("float", [None, 2])

    p_keep_input = tf.placeholder("float")
    p_keep_hidden = tf.placeholder("float")

    w_h1 = init_weights([4554, 221])
    w_h2 = init_weights([221, 21])
    # w_h3 = init_weights([200,30])
    w_o = init_weights([21, 2])

    # b_h1 = init_weights([211])
    # b_h2 = init_weights([21])
    # b_o = init_weights([2])

    X = tf.nn.dropout(X,p_keep_input)

    h1 = tf.nn.sigmoid(tf.matmul(X, w_h1),"h1")
    h1 = tf.nn.dropout(h1,p_keep_hidden)

    h2 = tf.nn.sigmoid(tf.matmul(h1, w_h2),"h2")
    h2 = tf.nn.dropout(h2,p_keep_hidden)

    # h3 = tf.nn.sigmoid(tf.matmul(h2,w_h3))
    # h3 = tf.nn.dropout(h3,p_keep_hidden)

    py_x = tf.matmul(h2, w_o)

    beta = 0.03
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y)
                          +beta*(tf.nn.l2_loss(w_h1)+tf.nn.l2_loss(w_h1)+tf.nn.l2_loss(w_o)))
    #train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
    #train_op = tf.train.RMSPropOptimizer(0.011, 0.9).minimize(cost)
    train_op = tf.train.AdadeltaOptimizer(0.001,0.9).minimize(cost)

    predict_op = tf.argmax(py_x, 1)

    correct_pred = tf.equal(tf.argmax(Y, 1), tf.argmax(py_x, 1))  # Count correct predictions
    acc_op = tf.reduce_mean(tf.cast(correct_pred, "float"))  # Cast boolean to float to average

    #for visualize
    tf.histogram_summary("w_h1",w_h1)
    tf.histogram_summary("w_h2",w_h2)
    tf.histogram_summary("w_o",w_o)
    tf.scalar_summary("cost",cost)
    tf.scalar_summary("accuracy",acc_op)


    print("Begin to train model.")
    with tf.Session() as sess:

        file_dir = "./log/mlp_logs_" + time.strftime("%a-%d-%b-%Y--%H:%M:%S", time.localtime())
        writer = tf.train.SummaryWriter(file_dir,sess.graph)
        merged = tf.merge_all_summaries()

        tf.initialize_all_variables().run()


        print("init varialbles.")
        for i in range(1000):
            # for start,end in zip(range(0,x.shape[0],100),range(100,x.shape[0],100)):
            #     sess.run(train_op,feed_dict = {X:x[start:end],Y:y[start:end]})
            print("train")
            #7202
            train_N = 20
            batch_size = 200
            for j in range(1, train_N):
                #print("data from : ", (i - 1) * 50, " to ", i * 50)
                sess.run(train_op, feed_dict={X: x2[(j - 1) * batch_size+1:j * batch_size,:],
                                              Y: y2[(j- 1) * batch_size+1:j * batch_size,:]
                                            ,p_keep_input: 0.95, p_keep_hidden: 0.95})

            summary , acc = sess.run([merged,acc_op],
                                     feed_dict={X: x, Y: y
                                         , p_keep_input: 1.0, p_keep_hidden: 1.0})

            writer.add_summary(summary,i)

            print ("test",i, np.mean(np.argmax(y[batch_size*train_N::], axis=1) ==
                              sess.run(predict_op, feed_dict={X: x[batch_size*train_N::,:], Y: y[batch_size*train_N::]
                                  , p_keep_input: 1.0, p_keep_hidden: 1.0})))
            print("train",i, np.mean(np.argmax(y[0:batch_size*train_N], axis=1) ==
                             sess.run(predict_op, feed_dict={X: x[0:batch_size*train_N, :], Y: y[0:batch_size*train_N]
                                 , p_keep_input: 1.0, p_keep_hidden: 1.0})))

            #Best````
            # test
            # 856
            # 0.921611492817
            # train
            # 856
            # 0.99425