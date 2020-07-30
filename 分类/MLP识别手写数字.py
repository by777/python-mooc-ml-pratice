# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 21:06:22 2020

@author: Xu Bai
数字手写字体识别
数据集：DBTHD MNIST
MNIST ： 28 * 28 train / test = 60000 / 10000
DBRHD : 32 * 32 7494 (40位手写者)/ 3498（14位手写者） 

常用于数字手写体的分类器：
+ 线性分类器
+ KNN
+ 非线性分类器
+ SVM
+ MLP多层感知机
+ CNN
"""
import numpy as np
from os import listdir
from sklearn.neural_network import MLPClassifier

# 将加载的32 * 32的图片矩阵展开位一列向量
def img2vector(fileName):
    # 定义返回的矩阵大小为 1 * 1024
    retMat = np.zeros([1024],int)
    fr = open(fileName)
    lines = fr.readlines()
    for i in range(32):
        for j in range(32):
            retMat[i * 32 + j] = lines[i][j]
    return retMat

def readDataSet(path):
    fileList = listdir(path)
    numFiles = len(fileList)
    # 用于存放所有的数字文件
    dataSet = np.zeros([numFiles,1024],int)
    # 用于存放one-hot
    hwLabels = np.zeros([numFiles,10])
    # 遍历所有文件
    for i in range(numFiles):
        # 获取文件名称 / 路径
        filePath = fileList[i]
        # 通过文件名获取标签
        digit = int(filePath.split('_')[0])
        # 将对应的one-hot置1
        hwLabels[i][digit] = 1.0
        dataSet[i] = img2vector(path + '/' + filePath)
    return dataSet,hwLabels

train_dataSet, train_hwLabels = readDataSet('trainingDigits')
# 构造神经网络
clf = MLPClassifier(hidden_layer_sizes=(100,),activation='logistic',solver='adam',
                    learning_rate_init=0.0001,max_iter=2000)
print(clf)
clf.fit(train_dataSet,train_hwLabels)
# 测试
dataSet,hwLabels = readDataSet('testDigits')
res = clf.predict(dataSet)   #对测试集进行预测
error_num = 0                #统计预测错误的数目
num = len(dataSet)           #测试集的数目
#遍历预测结果
for i in range(num):
    #比较长度为10的数组，返回包含01的数组，0为不同，1为相同
    #若预测结果与真实结果相同，则10个数字全为1，否则不全为1
    if np.sum(res[i] == hwLabels[i]) < 10: 
        error_num += 1
print("Total num:",num," Wrong num:", \
      error_num,"  WrongRate:",error_num / float(num))