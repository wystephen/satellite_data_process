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
    clf.fit(x[1:3000,:],y[1:3000])

    pre_y = clf.predict(x[3000::])

    err = np.zeros_like(pre_y)
    np.abs(pre_y-y[3000::],err)

    plt.figure(1)
    plt.hist(err)
    plt.show()



