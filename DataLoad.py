# -*- coding:utf-8 -*-
# Create by steve in 16-9-6 at 下午9:20

import numpy as np
import random as rd

class DataLoad:
    def __init__(self):
        #self.oldX = np.loadtxt("outX.txt")
        #self.oldY = np.loadtxt("outY.txt")

        self.Xpos = np.loadtxt("Data/Xpos.txt")
        self.Xneg = np.loadtxt("Data/Xneg.txt")

        self.Ypos = np.loadtxt("Data/Ypos.txt")
        self.Yneg = np.loadtxt("Data/Yneg.txt")



        rd.seed()

    def toOneDataset(self):
        self.X = np.zeros([self.Xpos.shape[0]+self.Xneg.shape[0],self.Xpos.shape[1]])
        self.Y = np.zeros([self.Ypos.shape[0]+self.Yneg.shape[0]])

        #self.X = np.zeros([self.oldX.shape[1]])
        #self.Y = np.zeros([self.oldY.shape[1]])
        self.X[0:self.Xpos.shape[0],:] = self.Xpos
        self.X[self.Xpos.shape[0]::,:] = self.Xneg

        self.Y[0:self.Ypos.shape[0]] = self.Ypos
        self.Y[self.Ypos.shape[0]::] = self.Yneg

        print(self.X.shape,self.Y.shape)


    def random_sample(self,N=1000):
        


    def nomalized_X(self,X):
        mean = X.mean()
        std = X.std()

        for i in range(X.shape[0]):
            X[i,:] = X[i,:]-mean
            X[i,:] = X[i,:] / mean



if __name__ == '__main__':
    dl = DataLoad()
    dl.toOneDataset()

