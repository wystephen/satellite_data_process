# -*- coding:utf-8 -*-
# Create by steve in 16-9-6 at 下午9:20

import numpy as np
import random as rd

import matplotlib.pyplot as plt

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
        tmp_x = np.zeros([self.X.shape[1]])
        tmp_y = np.zeros([1])
        for j in range(N):
            a = rd.randrange(0,self.X.shape[0])
            b = rd.randrange(0,self.X.shape[0])

            if j%100==0:
                print(j,N)

            if a==b:
                continue
            else:
                tmp_x = self.X[b,:]
                tmp_y = self.Y[b]
                self.X[b,:] = self.X[a,:]
                self.Y[b] = self.Y[a]

                self.X[a,:] = tmp_x
                self.Y[a] = tmp_y

    #Normalize the
    def nomalize_data(self):
        mean = np.zeros([self.X.shape[1]])
        std = np.zeros([self.X.shape[1]])

        self.X.mean(axis=0,out=mean)
        self.X.std(axis=0,out=std)

        for i in range(self.X.shape[0]):
            #self.X[i,:] = (self.X[i,:]-mean)/std
            for j in range(self.X.shape[1]):
                self.X[i,j] = (self.X[i,j] - mean[j])/std[j]
            if self.Y[i] <1.0:
                self.Y[i] = 0
            else:
                self.Y[i] = 1

        #for test
        print(self.X.mean(axis=0),self.X.std(axis=0))



    def data_save(self):
        np.savetxt("sumX.txt",self.X)
        np.savetxt("sunY.txt",self.Y)


    def nomalized_X(self,X):
        mean = X.mean()
        std = X.std()

        for i in range(X.shape[0]):
            X[i,:] = X[i,:]-mean
            X[i,:] = X[i,:] / mean


    def display(self):
        plt.figure("1")
        plt.plot(self.Y)

        plt.figure("2")
        plt.hist(self.X)

        plt.show()

if __name__ == '__main__':
    dl = DataLoad()
    dl.toOneDataset()
    dl.random_sample(4000)
    dl.nomalize_data()
    dl.data_save()
    #dl.display()
