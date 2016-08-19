# -*- coding:utf-8 -*-
# carete by steve at  2016 / 08 / 15　20:35

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

from sklearn import svm

from DataManage import DataManage

if __name__ == '__main__':
    dm = DataManage()
    x,y = dm.get_train_set()

    clf = svm.LinearSVC()
    train_size = 5000
    clf.fit(x[1:train_size,:],y[1:train_size])

    pre_y = clf.predict(x[train_size::])

    err = np.zeros_like(pre_y)
    np.abs(pre_y-y[train_size::],err)

    plt.figure(1)
    plt.hist(err)
    plt.show()



