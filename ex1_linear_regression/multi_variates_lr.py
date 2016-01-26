#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from common_widget.data_process import *
from common_widget.calculate import *


def feature_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = my_std(X)  # python 的std分母除的是 n，应该是 n-1
    for index, isimple in enumerate(X):
        X[index, :] = (isimple - mu) / sigma
    return X, mu, sigma


# ---计算梯度下降，给定迭代次数，输出最后得到的theta和每次迭代后的J_cost ---
def gradient_descent_multi(X, y, theta, alpha, num_iters):
    # X m行，n+1列; y-向量，python里的向量都是行向量（一维数组1*m）
    m = len(X)
    J_history = np.empty(num_iters)

    for i in range(num_iters):
        part_d = np.dot((np.dot(theta, X.T) - y), X) / m
        theta = theta - alpha * part_d
        J_history[i] = compute_cost_linear(X, y, theta)

    return theta, J_history


def predict(X, theta, pre_data, mu=None, sigma=None):
    # 要进行均值、方差修正
    if mu is not None:
        for index, isimple in enumerate(pre_data):
            pre_data[index, :] = (isimple - mu) / sigma

    # print pre_data
    #添加1列
    pre_data = np.concatenate((np.ones((len(pre_data), 1)), pre_data), axis=1)
    pre_result = np.dot(theta, pre_data.T)
    return pre_result


if __name__ == '__main__':
    # 数据读入，显示
    file_name = 'ex1data2.txt'
    X, y = get_data(file_name)
    print "打印3个样本："
    for i in range(3):
        print("x = [%.0f %.0f], y = %.0f" % (X[i, 0], X[i, 1], y[i]))

    # ---1) 特征缩放，防止某些特征值太大影响比重太大--
    # 归一化到0均值，标准差为1  (x<--(x-mu)/sigma )
    X, mu, sigma = feature_normalize(X)     # 均值0，标准差1
    print "mu: ", mu, "sigma: ", sigma

    # X添加1列
    X = np.concatenate((np.ones((len(X), 1)), X), axis=1)
    # print "检验X前3行", X[:3, :]

    # ---2) 梯度下降，得到最终的theta---
    alpha = 0.01
    num_iters = 8500
    theta = np.zeros(len(X[0]))
    theta, J_history = gradient_descent_multi(X, y, theta, alpha, num_iters)

    # ---画出梯度下降时，代价函数变化曲线---
    plt.figure(1)
    plt.plot(J_history)
    plt.title("the convergence graph")
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost J")
    plt.show()

    print "梯度下降得到的 theta= ", theta

    # ---3）预测--------------
    pre_data = np.array([[1650, 3], [2000, 7]], dtype=float)
    pre_result = predict(X, theta, pre_data, mu, sigma)
    print "预测的结果：1650, 3=>",pre_result[0], " 2000, 7=>", pre_result[1]

    # =======Normal Equations===========
    # 正则方程方式，不需要对特征进行缩放！
    X, y = get_data(file_name)
    pre_data = np.array([[1650, 3], [2000, 7]], dtype=float)
    # X添加1列
    X = np.concatenate((np.ones((len(X), 1)), X), axis=1)

    # --计算 theta = (X.T*X)`*X.T*y
    from numpy.linalg import pinv
    y_t = y.reshape(len(y), 1)
    theta = pinv(X.T.dot(X)).dot(X.T).dot(y_t).flatten()
    print "正则方程方式得到的 theta= ", theta
    # --预测
    pre_result = predict(X, theta, pre_data)
    print "预测的结果：1650, 3=>",pre_result[0], " 2000, 7=>", pre_result[1]