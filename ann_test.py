# -*- coding:utf-8 -*-
# carete by steve at  2016 / 08 / 19　17:45

import numpy as np

import matplotlib.pyplot as plt

from DataManage import  DataManage

from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

from pybrain.datasets import SupervisedDataSet

from pybrain.utilities import percentError



if __name__ == '__main__':
    dm = DataManage()
    x,y = dm.get_train_set()

    net = buildNetwork(4554,22,1,bias = True)
    ds = SupervisedDataSet(4554,1)
    t_ds = SupervisedDataSet(4554,1)
    max_iterations= 1000
    err_list = np.zeros([max_iterations])


    print "Begin set dataset"

    train_size = 5000
    for i in range(0,train_size-1):
        ds.addSample(x[i,:],y[i])

    for i in range(train_size,x.shape[0]):
        t_ds.addSample(x[i,:],y[i])


    print "Dataset set succesful,begin to test train"

    trainer = BackpropTrainer(net, ds)
    trainer.train()
    print "Test train ok,begin to train until convergence"

    for it in range(0,max_iterations):
        trainer.trainEpochs()
        print "epoch:",it
        if it % 10 ==0:
            print "accuracy rating of training dataset:"
            print percentError( trainer.testOnClassData(),ds)
        elif it % 20 == 0:
            print "accuracy ration of test dataset:"
            print percentError(trainer.testOnClassData(t_ds),t_ds)


    #trainer.trainUntilConvergence(maxEpochs=1000,continueEpochs=12)
    print "Network is convergenced."



