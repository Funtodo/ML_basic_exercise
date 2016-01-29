#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

# test = np.mat([[1,2,3],[4,5,6]])
# test = np.array(test)
# print test
# print test[0,:], test[1,:]
plt.figure(1)
plt.figure(2)
ax1 = plt.subplot(2,2,1) # 在图表2中创建子图1
ax2 = plt.subplot(2,2,2) # 在图表2中创建子图2
ax3 = plt.subplot(2,1,2) # 在图表2中创建子图2

x = np.linspace(0, 3, 100)
for i in xrange(5):
    plt.figure(1)  #? # 选择图表1
    plt.plot(x, np.exp(i*x/3))
    plt.sca(ax1)   #? # 选择图表2的子图1
    plt.plot(x, np.sin(i*x))
    plt.sca(ax2)  # 选择图表2的子图2
    plt.plot(x, np.cos(i*x))
    plt.sca(ax3)  # 选择图表2的子图2
    plt.plot(x, np.cos(i*x))

plt.show()