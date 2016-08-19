# -*- coding:utf-8 -*-
# carete by steve at  2016 / 08 / 19　17:45

import numpy as np

import matplotlib.pyplot as plt

from DataManage import  DataManage

from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

from pybrain.datasets import SupervisedDataSet


if __name__ == '__main__':
    dm = DataManage()
    x,y = dm.get_train_set()

    net = buildNetwork(4554,22,1,bias = True)
    ds = SupervisedDataSet(4554,1)

    print "Begin set dataset"

    train_size = 5000
    for i in range(0,train_size-1):
        ds.addSample(x[i,:],y[i])

    print "Dataset set succesful,begin to test train"

    trainer = BackpropTrainer(net, ds)
    trainer.train()
    print "Test train ok,begin to train until convergence"
    trainer.trainUntilConvergence(maxEpochs=1000,continueEpochs=12)
    print "Network is convergenced."

    pre_y = np.zeros(x.shape(0)-train_size)
    for k in range(0,(pre_y.shape(0)-1)):
        pre_y[k] = net.active(x[train_size+k,:])
    print "Predict over."

    err = np.zeros_like(pre_y)
    np.abs(pre_y-y[train_size::],err)

    plt.figure(1)
    plt.hist(err)
    plt.show()