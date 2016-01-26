#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
# from common_widget import data_process, calculate
from common_widget.data_process import *
from common_widget.calculate import *


def gradient_descent(X, y, theta, alpha, iterations):
    # print "X size: ", X.shape, "type: ", X.dtype
    # print "theta size: ", theta.shape, "type: ", theta.dtype
    # print "Y size: ", y.shape, "type: ", y.dtype
    for i in range(iterations):
        par_d = np.dot(X.T, np.dot(X, theta) - y)/m
        theta = theta - alpha * par_d

    return theta


if __name__ == '__main__':

    # --数据读入，显示--
    file_name = 'ex1data1.txt'
    X, y = get_data(file_name)
    m = len(y)  # 样本数
    plt.figure(1)
    plt.plot(X, y, 'rx')
    plt.title("origin_data")
    plt.show()

    # --添加一列1给X
    # X = add_one_col(X)
    # print "X size: ", X.shape, "type: ", X.dtype
    # print "y size: ", y.shape, "type: ", y.dtype
    X = np.concatenate((np.ones_like(X), X), axis=1) # 此处X：m*1
    print "new X size: ", X.shape, "type: ", X.dtype

    # 初始化theta
    theta = np.zeros(2)
    # 初始化学习率 alpha, 迭代次数
    alpha = 0.01
    iterations = 1500

    # 计算代价函数
    J = compute_cost_linear(X, y, theta)
    print "初始化时，J = ", J

    # --梯度下降--
    theta = gradient_descent(X, y, theta, alpha, iterations)
    print "梯度下降得到的 theta= ", theta

    # --画出线性拟合
    plt.figure(2)
    plt.plot(X[:, 1], y, 'rx', label='Training data')
    plt.plot(X[:, 1], np.dot(X, theta), 'g-', label='Linear regression')
    plt.title("univariate_linear_regression")
    plt.legend()
    plt.show()

    # --预测--
    pre_data = np.array([[1, 3.5], [1, 7]])  # 直接有添加1列
    pre_result = np.dot(pre_data, theta)
    print "预测的结果：3.5=>",pre_result[0], " 7=>", pre_result[1]

    # ---可视化代价函数 J---
    # 得到计算J的theta取值
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    # 初始化 J_vals 矩阵
    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

    # 计算 J_vals
    for i in range(len(theta0_vals)):
        for j in range(len(theta1_vals)):
            tmp_theta = np.array([theta0_vals[i], theta1_vals[j]])
            J_vals[i, j] = compute_cost_linear(X, y, tmp_theta)

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)  #
    ax.plot_surface(theta0_vals, theta1_vals, J_vals, rstride=1, cstride=1, cmap='hot')
    ax.contourf(theta0_vals, theta1_vals, J_vals, zdir='z', offset=-2, cmap=plt.cm.hot)
    ax.set_xlabel("theta0")
    ax.set_ylabel("theta1")
    ax.set_zlabel("J_vals")
    plt.show()
