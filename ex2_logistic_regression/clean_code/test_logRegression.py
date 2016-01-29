#!/usr/bin/python
# -*- coding: utf-8 -*-

##############################################
# 测试逻辑回归(logRegression)
# Author: Jessie
# Date: 2016-01-28
# Referenced blog： http://blog.csdn.net/zouxy09
##############################################

from logRegression import *
import numpy as np

# 加载数据，并给X添加1列，返回矩阵
def loadData():
    txtData = np.loadtxt('ex2data1.txt', delimiter=',')  #自动类型转换识别/转换，float
    train_X = txtData[:, :2]  # 前两列是X的特征
    train_y = txtData[:, 2]  #最后一列是y，类标
    # X添加一列 1
    train_X = np.concatenate((np.ones((len(train_X), 1)), train_X), axis=1)  # 在列方向上合并，axis = 1
    return np.mat(train_X), np.mat(train_y).T


# 加载数据，一行行读数据，处理
def loadData_way2():
    train_X = []
    train_y = []
    file = open('testDataSet.txt')
    for line in file.readlines():
        lineArr = line.strip().split()
        train_X.append([1.0, lineArr[0], lineArr[1]])  #在每行append时就添加1列，1.0--防止变成整形
        train_y.append(lineArr[2])
    return np.mat(train_X), np.mat(train_y).T

''' ======清晰的代码框架很重要===== '''

# # ======步骤1：加载数据=============
print "step 1: Load data..."
# 加载数据时，已将数据转换为矩阵 matrix，并且X添加了一列1（X：m*(n+1),,,y: m*1）
train_X, train_y = loadData()
# train_X, train_y = loadData_way2()
test_X, test_y = train_X, train_y  # 暂且用训练集做测试集


# # ======步骤2：训练================
print "step 2: training..."
''' ----将模型训练参数用字典给出----'''
opts = {'alpha': 0.1, 'maxIter': 3000, 'optimizeType': 'gradDescent'}
optimalWeights, weightsHist = trainLogRegres(train_X, train_y, opts)
print optimalWeights

# #======补充步骤，输出迭代损失曲线，以及迭代各参数变化曲线
# 调试，观察模型用
print "observe model status: "
observeModel(train_X, train_y, weightsHist)


# # ======步骤3：测试================
print "step 3: testing..."
accuracy = testLogRegres(optimalWeights, test_X, test_y)

# # ======步骤4：展示结果============
print "step4: show the result..."
print 'The classify accuracy is %.3f%%' % (accuracy*100)
showLogRegres(optimalWeights, train_X, train_y)
