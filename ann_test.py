# -*- coding:utf-8 -*-
# carete by steve at  2016 / 08 / 19　17:45

import numpy as np

import matplotlib.pyplot as plt

from DataManage import DataManage

from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

from pybrain.datasets import SupervisedDataSet
from pybrain.datasets import ClassificationDataSet

from pybrain.utilities import percentError

from pybrain.tools.neuralnets import NNclassifier

if __name__ == '__main__':
    dm = DataManage()
    x, y = dm.get_train_set()

    net = buildNetwork(4554, 22, 2, bias=True)
    ds = ClassificationDataSet(4554, 1, nb_classes=2)
    t_ds = ClassificationDataSet(4554, 1, nb_classes=2)
    max_iterations = 50000
    err_list = np.zeros([max_iterations])

    print "Begin set dataset"

    train_size = 5000
    for i in range(0, train_size - 1):
        ds.addSample(x[i, :], y[i])

    for i in range(train_size, x.shape[0]):
        t_ds.addSample(x[i, :], y[i])

    ds._convertToOneOfMany()
    t_ds._convertToOneOfMany()

    print "Dataset set succesful,begin to test train"

    trainer = BackpropTrainer(net, ds)
    trainer.train()
    print "Test train ok,begin to train until convergence"

    train_accuracy = 0.0
    valid_accuracy = 0.0
    min_err = 100.0
    min_err_train = 100.0

    for it in range(0, max_iterations):
        trainer.trainEpochs()
        print "epoch:", it
        #if it % 10 == 0:
            #print "accuracy rating of training dataset:", \
             #   percentError(trainer.testOnClassData(), ds['class'])
        #if it % 20 == 0:
            #print "accuracy ration of test dataset:", \
            #    percentError(trainer.testOnClassData(t_ds), t_ds['class'])
        if it % 5 == 0:
            err = percentError(trainer.testOnClassData(t_ds), t_ds['class'])
            err_train = percentError(trainer.testOnClassData(),ds['class'])
            if err_train < min_err_train:
                min_err_train = err_train
                print "min err of train dataset is:",min_err_train
            if err < min_err:
                min_err = err
                print "min err of valid datset is:", min_err


    # trainer.trainUntilConvergence(maxEpochs=1000,continueEpochs=12)
    print "Network is convergenced."
    print "min err of valid datset is:",min_err
