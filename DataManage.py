# -*- coding:utf-8 -*-
# carete by steve at  2016 / 08 / 15　20:10

import scipy as sp
import numpy as np

#import theano
#import theano.tensor as T
#import theano.configdefaults

import logistic_cg
import logistic_sgd

# __float__ = 'float32'

class DataManage:
    def __init__(self):
        self.x = np.loadtxt("outX.txt")
        self.y = np.loadtxt("outY.txt")

    def get_train_set(self):
        return self.x, self.y

    def pca(self):
        print 'pca'
    #
    # def shared_data(X, Y, self):
    #
    #     shared_x = theano.shared(np.asarray(X,dtype=theano.config.floatX), borrow=True)
    #     shared_y = theano.shared(np.asarray(Y,dtype=theano.config.floatX), borrow=True)
    #
    #     return shared_x, T.cast(shared_y, 'int32')

    # def theano_type_data(self, train_precent=0.5, valid_precent=0.3):
    #     if train_precent + valid_precent > 1.0:
    #         print (u"train dataset 和 valid dataset 占比必须小于一")
    #     train_index = int(self.x.shape[0]* train_precent)
    #     valid_index = int(self.x.shape[0] * (train_precent + valid_precent))
    #
    #     train_x, train_y = self.shared_data(self.x[1:train_index, :], self.y[1:train_index])
    #     test_x, test_y = self.shared_data(self.x[train_index:valid_index, :], self.y[train_index:valid_index])
    #     valid_x, valid_y = self.shared_data(self.x[(train_index + valid_index)::],
    #                                         self.y[(train_index + valid_index)::])
    #     return train_x, train_y, test_x, test_y, valid_x, valid_y


if __name__ == '__main__':
    dm = DataManage()
    x, y = dm.get_train_set()

    print x.shape
    #
    #
    # train_x,train_y,test_x,test_y,valid_x,valid_y = dm.theano_type_data()
