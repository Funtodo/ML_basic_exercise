#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import numpy as np

# ---获得数据，适用于数据文件在当前目录下，且以逗号分隔，最后一列为y----
def get_data(file_name):
    # BASE_DIR = os.path.dirname(__file__)
    pwd = os.getcwd()
    url = os.path.join(pwd, file_name)
    odata = np.loadtxt(url, delimiter=',')
    X = odata[:, :len(odata[0])-1]  # 最后一列为y，即使X只有一个特征，得到的也是二维数组（m*1）
    y = odata[:, -1]  # 指定只有一列，则得到的y是一维数组，在python里，一维数组相当于行向量(最后一列可用-1索引)
    return X, y


# ---给样本X矩阵添加一列1---
# def add_one_col(X):
#     X_old = X
#     X = np.empty((len(X), len(X[0])+1))
#     X[:, 0] = np.ones(len(X))
#     X[:, 1:] = X_old
#     return X



# 测试git
# 测试git
# 测试git
# 测试git
# 测试git
# 测试git