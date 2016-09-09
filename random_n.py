# -*- coding:utf-8 -*-
# Create by steve in 16-9-9 at 下午7:21

import numpy as np
import random as rd

import matplotlib.pyplot as plt

if __name__ == '__main__':
    x = np.loadtxt('norX.txt')
    y = np.loadtxt('norY.txt')

    times = 200

    tmpx = np.zeros(x.shape[1])
    tmpy = np.zeros(1)

    for i in range(times):
        index1 = int(rd.randrange(1,x.shape[0]-2))
        index2 = int(rd.randrange(1, x.shape[0] - 2))

        if index1 == index2:
            continue
        else:
            tmpx = x[index1,:]
            tmpy = y[index1]

            x[index1,:] = x[index2,:]
            y[index1]  = y[index2]

            x[index2,:] = tmpx
            y[index2] = tmpy

    np.savetxt('norXrnd.txt',x)
    np.savetxt('norYrnd.txt',y)
