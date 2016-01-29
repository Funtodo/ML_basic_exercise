#!/usr/bin/python
# -*- coding: utf-8 -*-

##############################################
# 逻辑回归模块化(logRegression)
# Author: Jessie
# Date: 2016-01-28
# Referenced blog： http://blog.csdn.net/zouxy09
# 函数概览：trainLogRegres--训练逻辑回归模型
#          testLogRegres--测试
#          showLogRegres--图形化展示结果
##############################################

import numpy as np


# 辅助函数　##
def sigmoid(X):
    X = 1.0 / (1 + np.exp(-1 * X))
    return X

# 训练逻辑回归模型，并提供可选优化算法（三种不同的梯度下降）
# input: train_X, 样本特征，矩阵形式(mat)，m*n，每行代表一个样本(已包含1列)
#        train_y，样本标签，矩阵形式(mat)，m*1，每行代表一个样本标签
#        opts，模型训练参数，字典形式(dict)，例如，opts = {'alpha': 0.01, 'maxIter': 20, 'optimizeType': 'gradDescent'}
# output: weights，得到的模型参数优化结果，矩阵形式，n*1，
#         weights_hist，得到的权重历史（用作模型分析）
def trainLogRegres(train_X, train_y, opts):

    numSamples, numFeatures = np.shape(train_X)
    alpha = opts['alpha']
    maxIter = opts['maxIter']
    # 初始化权重矩阵，n*1全0矩阵(注意全部用float型)
    # 不然会报错：Cannot cast ufunc subtract output from dtype('float64') to dtype('int32') with casting rule 'same_kind'
    weights = np.mat([[0.0]]*numFeatures)

    if opts['optimizeType'] == 'gradDescent':  # 批处理梯度下降
        weightsHist = np.asmatrix(np.empty((numFeatures, maxIter+1)))  # 每一列代表一个weight历史
        weightsHist[:, 0] = weights
        for i in range(maxIter):
            weights -= alpha * (train_X.T * (sigmoid(train_X*weights)-train_y))
            weightsHist[:, i+1] = weights

    elif opts['optimizeType'] == 'stocDescent':  # 简单随机梯度下降
        weightsHist = np.asmatrix(np.empty((numFeatures, maxIter*numSamples+1)))  # 每一列代表一个weight历史
        weightsHist[:, 0] = weights
        for i in range(maxIter):
            for j in range(numSamples):  # 对每个样本进行迭代
                weights -= alpha * ((sigmoid(train_X[j, :]*weights)-train_y[j])*train_X[j, :]).T
                weightsHist[:, i*numSamples+j+1] = weights

    elif opts['optimizeType'] == 'smoothStocGradDescent':  # 优化随机梯度下降：alpha随着迭代次数增加而变小，且每次迭代随机样本顺序
        import random
        weightsHist = np.asmatrix(np.empty((numFeatures, maxIter*numSamples+1)))  # 每一列代表一个weight历史
        weightsHist[:, 0] = weights
        for i in range(maxIter):
            randIndex = range(numSamples)
            random.shuffle(randIndex)
            for j in randIndex:
                weights -= alpha * ((sigmoid(train_X[j, :]*weights)-train_y[j])*train_X[j, :]).T
                weightsHist[:, i*numSamples+j+1+1] = weights
            if alpha > 0.00001:  # alpha不至于变得太小，接近于0就没有意义了
                alpha *= 0.9
    else:
        print "optimizeType parameter is error!"

    return weights, weightsHist
    # return weights


# 计算模型预测精度
# input: optimalWeights-获得的最优参数-n*1; test_X, mat, m*n; test_y, mat, m*1
# output: 计算精度，小数
def testLogRegres(optimalWeights, test_X, test_y):
    pred_hx = sigmoid(test_X * optimalWeights)
    accuracy = np.mean((pred_hx >= 0.5) == test_y)
    return accuracy


# 图形化展示逻辑回归（决策边界）-仅当X含二维特征时
def showLogRegres(optimalWeights, train_X, train_y):
    # --注意： train_X, train_y 为矩阵类型（mat datatype）
    numSamples, numFeatures = np.shape(train_X)
    if numFeatures != 3:  # 包含一维1
        print "无法进行图形化展示，只有二维特征时可以；"
        return 1

    import matplotlib.pyplot as plt
    plt.figure()
    # 矩阵的 bool切片似乎不能实现，，并且也没有将mat转为array的函数
    # for 循环好了，，
    for idx, ilable in enumerate(train_y):
        if ilable == 1:
            plt.plot(train_X[idx, 1], train_X[idx, 2], '+k')
        else:
            plt.plot(train_X[idx, 1], train_X[idx, 2], 'ob')

    # min, max 前面去掉np.，得到的不是一个数，而是一个矩阵，，要用numpy才是得到的数
    ax_x1 = np.array([np.min(train_X[:, 1]), np.max(train_X[:, 1])])

    # 不能直接写optimalWeights[0]（得到的是一个1*1矩阵，不是数），虽然其为向量
    ax_x2 = -(optimalWeights[0, 0]+optimalWeights[1, 0]*ax_x1) / optimalWeights[2, 0]
    plt.plot(ax_x1, ax_x2, '-r', label='Decision Boundary')
    plt.show()


# weightsHist n*numIters 每一列表示一个 迭代历史 参数
def observeModel(train_X, train_y, weightsHist):

    import matplotlib.pyplot as plt
    plt.figure()
    ax_x = np.arange(len(weightsHist.T))
    plt.subplot(4, 1, 2)  # 画第一个子图--损失函数变化趋势
    plt.title('lost function')
    prob_hx = sigmoid(train_X*weightsHist)  # mat m * numIters
    test = np.log(prob_hx)
    # lost_values = train_y.T * np.log(prob_hx) + (1-train_y).T * np.log(prob_hx-train_y)  # mat 1*numIters
    # 运算量可能太大，，换循环试试，，，


    lost_values = -1 / len(train_y) * lost_values
    lost_values = np.array(lost_values).flatten()
    plt.plot(ax_x, lost_values)

    weightsHist = np.array(weightsHist)  # 转成数组，方便画图
    plt.subplot(4, 1, 2)  # 画第二个子图--weight_0
    plt.title('weight_0')
    plt.plot(ax_x, weightsHist[0, :])
    plt.subplot(4, 1, 3)  # 画第三个子图--weight_1
    plt.title('weight_1')
    plt.plot(ax_x, weightsHist[1, :])
    plt.subplot(4, 1, 4)
    plt.title('weight_2')
    plt.plot(ax_x, weightsHist[2, :])
    plt.show()
