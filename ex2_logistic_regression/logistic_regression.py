#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from common_widget.data_process import *
from common_widget.calculate import *
import matplotlib.pyplot as plt
from math import log, exp

def feature_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = my_std(X)  # python 的std分母除的是 n，应该是 n-1
    for index, isimple in enumerate(X):
        X[index, :] = (isimple - mu) / sigma
    return X, mu, sigma


def plot_log_reg_traindata(X, y):
    pos_sample = X[y == 1, :]
    neg_sample = X[y == 0, :]
    # print X.shape, pos_sample.shape, neg_sample.shape
    plt.figure()
    plt.plot(pos_sample[:, 0], pos_sample[:, 1], 'b+', markersize=10, label='positive')
    plt.plot(neg_sample[:, 0], neg_sample[:, 1], 'yo', markersize=7, label='negative')
    plt.legend()
    plt.show()


def sigmoid(z):
    # 对一维数组每个元素z，计算 1/(1+e^-z)
    sigmoid_ufunc = np.frompyfunc(lambda x: 1 / (1 + exp(-1 * x)), 1, 1)
    z = sigmoid_ufunc(z)
    # print z.shape, z.dtype
    '''也不行，，因为float也没有len（），，说明此函数只能对数组计算
    if len(z) != 1:  #说明只有一个数，只有一个数时，得到的不是object类，进行类型转换会报错
        z = z.astype(np.float64)  #进行类型转换
        '''
    return z.astype(np.float64)


def compute_cost_logistic(X, y, theta):
    m = len(X)
    # print X.shape, y.shape, theta.shape
    z = np.dot(theta, X.T)
    hx_predict = sigmoid(z)  # 预测值h(x)
    # print "hx_predict", hx_predict.shape, hx_predict.dtype
    # 报错了，，当hx_predict_i = 0/1 时，出现log0，非法计算---???程序有错才会出现！
    J = -1 * sum(y * np.log(hx_predict) + (1 - y) * np.log(1 - hx_predict)) / m
    return J

def gradient_descent_logistics(X, y, theta, alpha, num_iters):
    m = len(X)
    J_history = np.empty(num_iters+1)
    J_history[0] = 0
    for i in range(num_iters):
        # print "-----", i, "------------"
        hx = sigmoid(np.dot(theta, X.T))
        part_d = np.dot(hx-y, X) / m
        if i == 0:
            print "初始化的theta(0)对应的偏导: ", part_d
        theta -= alpha * part_d
        J_history[i+1] = compute_cost_logistic(X, y, theta)
    return theta, J_history

def plotDecisionBoundary(theta, X, y):
    pos_sample = X[y == 1, :]
    neg_sample = X[y == 0, :]
    # print X.shape, pos_sample.shape, neg_sample.shape
    plt.figure()
    plt.plot(pos_sample[:, 0], pos_sample[:, 1], 'b+', markersize=10, label='positive')
    plt.plot(neg_sample[:, 0], neg_sample[:, 1], 'yo', markersize=7, label='negative')
    x1 = np.array([min(X[:, 0]), max(X[:, 0])])  # 两点确定一条直线
    x2 = (-1/theta[2]) * (theta[0] + theta[1]*x1)
    plt.plot(x1, x2, 'r-', label='decisionBoundary')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    file_name = "ex2data1.txt"  # ex2data1.txt 常规梯度下降有问题
    # file_name = "test.txt"
    X, y = get_data(file_name)
    print X.dtype
    # X 是二维时，，可以画图表示---
    plot_log_reg_traindata(X, y)

    ''' ---!!很重要-看似数据一致，，但是不缩放，梯度下降时，
    alpha = 0.01时数据就层显震荡现象，且迭代下降缓慢，100’0000才能到0.224
    '''
    # ---特征缩放，防止某些特征值太大影响比重太大--
    # 归一化到0均值，标准差为1  (x<--(x-mu)/sigma )
    X, mu, sigma = feature_normalize(X)     # 均值0，标准差1
    print "mu: ", mu, "sigma: ", sigma

    m, n = X.shape
    # X添加1列
    X = np.concatenate((np.ones((len(X), 1)), X), axis=1)
    # 初始化theta
    theta = np.zeros(n + 1)
    # theta = np.array([ -15.39517866,   0.12825989,   0.12247929])
    # 初始化学习率 alpha, 迭代次数
    alpha = 0.1
    num_iters = 3000

    # ---计算代价函数----
    # print sigmoid(np.array([1,2,3]))
    J = compute_cost_logistic(X, y, theta)
    print "初始化时，J = ", J

    # ---进行梯度下降----
    theta, J_history = gradient_descent_logistics(X, y, theta, alpha, num_iters)
    J_history[0] = J
    print "梯度下降得到的 theta= ", theta
    print "得到的最小代价 J= ", J_history[-1]
    # ---画出梯度下降时，代价函数变化曲线---
    plt.figure(1)
    plt.plot(J_history)
    plt.title("the convergence graph")
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost J")
    plt.show()

    # 画出决策边界
    plotDecisionBoundary(theta, X[:, 1:], y)

    # 预测和精确度计算
    test_data = np.array([1, 45, 85], dtype=np.float64)
    ''' 注意修正均值，方差'''
    test_data[1:] = (test_data[1:] - mu) / sigma
    prob = 1 / (1 + exp(-1 * np.vdot(test_data, theta)))
    print "[45, 85]==》1 对应概率为： ", prob

    # 计算精度（预测训练集，>=0.5的预测为1）
    hx_prob = sigmoid(np.dot(theta, X.T))  # 预测值h(x)
    bool_prob = [x >= 0.5 for x in hx_prob]
    print "训练集精度：", np.mean(bool_prob==y) * 100
