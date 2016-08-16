# -*- coding:utf-8 -*-
# carete by steve at  2016 / 08 / 15　20:10

import scipy as sp
import numpy as np

import theano
import theano.tensor as T

class DataManage:
    def __init__(self):
        self.x = np.loadtxt("outX.txt")
        self.y = np.loadtxt("outY.txt")

    def get_train_set(self):
        return self.x,self.y

    def pca(self):
        print 'pca'

    def shared_data(X,Y,self):
        shared_x = theano.shared(np.asarray(X,dtype=theano.config.floatX),borrow=True)
        shared_y = theano.shared(np.asarray(Y,dtype=theano.config.floatX),borrow=True)

        return shared_x,T.cast(shared_y ,'int32')

    def theano_type_data(self,train_precent=0.5,valid_precent = 0.3):
        if train_precent + valid_precent > 1.0:
            print u"train dataset 和 valid dataset 占比必须小于一"
        






if __name__ == '__main__':
    dm = DataManage()
    x,y = dm.get_train_set()

    print x.shape
    print y.shape