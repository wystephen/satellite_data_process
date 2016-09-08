# -*- coding:utf-8 -*-
# Create by steve in 16-9-8 at 下午7:37

import tensorflow as tf

import numpy as np
import scipy as sp
import matplotlib as plt
import time


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

if __name__ == '__main__':
    #Load data
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

    #Set model
    x = x.reshape(-1,46,99,1)
    x2=x.reshape(-1,46,99,1)

    X = tf.placeholder("float",[None,46,99,1])
    Y = tf.placeholder("float",[None,2])

    w = init_weights([3,3,1,32])
    w2 = init_weights([3,3,32,64])
    w3 = init_weights([3,3,64,128])
    w4 = init_weights([128 * 6 *13 ,300])
    w_o = init_weights([300,2])

    p_keep_conv = tf.placeholder("float")
    p_keep_hidden = tf.placeholder("float")

    l1a = tf.nn.relu(tf.nn.conv2d(X, w,                       # l1a shape=(?, 46, 99, 32)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 23, 50, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,                     # l2a shape=(?, 23, 50, 64)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],              # l2 shape=(?, 12, 25, 64)
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,                     # l3a shape=(?, 12, 25, 128)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],              # l3 shape=(?, 6, 13, 128)
                        strides=[1, 2, 2, 1], padding='SAME')

    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    py_x = tf.matmul(l4,w_o)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
    train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
    # train_op = tf.train.RMSPropOptimizer(0.011, 0.9).minimize(cost)
    predict_op = tf.argmax(py_x, 1)

    correct_pred = tf.equal(tf.argmax(Y, 1), tf.argmax(py_x, 1))  # Count correct predictions
    acc_op = tf.reduce_mean(tf.cast(correct_pred, "float"))  # Cast boolean to float to average

    # for visualize
    tf.histogram_summary("w_h1", w)
    tf.histogram_summary("w_h2", w4)
    tf.histogram_summary("w_o", w_o)
    tf.scalar_summary("cost", cost)
    tf.scalar_summary("accuracy", acc_op)

    print("Begin to train model.")
    with tf.Session() as sess:

        file_dir = "./log/mlp_logs_" + time.strftime("%a-%d-%b-%Y--%H:%M:%S", time.localtime())
        writer = tf.train.SummaryWriter(file_dir, sess.graph)
        merged = tf.merge_all_summaries()

        tf.initialize_all_variables().run()

        print("init varialbles.")
        for i in range(1000):
            # for start,end in zip(range(0,x.shape[0],100),range(100,x.shape[0],100)):
            #     sess.run(train_op,feed_dict = {X:x[start:end],Y:y[start:end]})
            print("train")
            # 7202
            train_N = 20
            batch_size = 200
            for j in range(1, train_N):
                # print("data from : ", (i - 1) * 50, " to ", i * 50)
                sess.run(train_op, feed_dict={X: x2[(j - 1) * batch_size + 1:j * batch_size, :],
                                              Y: y2[(j - 1) * batch_size + 1:j * batch_size, :]
                    , p_keep_conv: 0.95, p_keep_hidden: 0.95})

            summary, acc = sess.run([merged, acc_op],
                                    feed_dict={X: x, Y: y
                                        , p_keep_conv: 1.0, p_keep_hidden: 1.0})

            writer.add_summary(summary, i)

            print ("test", i, np.mean(np.argmax(y[batch_size * train_N::], axis=1) ==
                                      sess.run(predict_op,
                                               feed_dict={X: x[batch_size * train_N::, :], Y: y[batch_size * train_N::]
                                                   , p_keep_conv: 1.0, p_keep_hidden: 1.0})))
            print("train", i, np.mean(np.argmax(y[0:batch_size * train_N], axis=1) ==
                                      sess.run(predict_op,
                                               feed_dict={X: x[0:batch_size * train_N, :], Y: y[0:batch_size * train_N]
                                                   , p_keep_conv: 1.0, p_keep_hidden: 1.0})))
