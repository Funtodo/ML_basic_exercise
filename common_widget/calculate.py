#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np


# ---计算方差（除以n-1版本）---
def my_std(X):
    mu = np.mean(X, axis=0)
    sigma = np.empty(len(X[0]))  # len(X[0])返回X的列数
    X = X.T  # X转置后，相当于求每行的标准差
    for index, isimple in enumerate(X):
        sigma[index] = (sum((isimple - mu[index]) ** 2) / (len(isimple) - 1)) ** 0.5
    return sigma

# ---计算代价函数（损失函数），适用于线性回归---
def compute_cost_linear(X, y, theta):
    m = len(y)
    J = sum((np.dot(theta, X.T) - y) ** 2) / (2 * m)
    return J
