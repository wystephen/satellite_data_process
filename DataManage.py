# -*- coding:utf-8 -*-
# carete by steve at  2016 / 08 / 15　20:10

import scipy as sp
import numpy as np

class DataManage:
    def __init__(self):
        self.x = np.loadtxt("outX.txt")
        self.y = np.loadtxt("outY.txt")

    def get_train_set(self):
        return self.x,self.y

    def pca(self):
        print 'pca'




if __name__ == '__main__':
    dm = DataManage()
    x,y = dm.get_train_set()

    print x.shape
    print y.shape