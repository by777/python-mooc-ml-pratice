# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 21:35:00 2020

@author: Xu Bai
--------------------------
PCA 主成分分析

主成分分析（Principal Component Analysis）是最常见的一种降维方法，
通常用于高维数据集的探索与可视化，还可以用作数据压缩和预处理等。

PCA可以把具有相关性的高维变量合成为线性无关的低维变量，称为主成分。
主成分能够尽可能保留原始数据的信息。

术语：
方差 / 协方差 / 协方差矩阵 / 特征向量和特征值
协方差：度量两个变量直接按的线性相关性程度，若两个变量的协方差为0，则可认为二者线性无关。
协方差矩阵是由变量的协方差值构成的矩阵（对称矩阵）
Cov(X,Y) =[ sum(X_i - X^-)(Y_i - Y^-) ] / ( n - 1)
---------------------------------------------------
原理：
矩阵的主成分就是其协方差矩阵对应的特征向量，
按照对应的特征值大小进行排序，最大的特征值就是第一主成分，以此类推

过程：
1.所有样本进行中心化
2.计算样本的协方差矩阵XX^T
3.对协方差矩阵XX^T做特征分解
4.取最大的几个特征值所对应的特征向量w
5.输出由上面的特征向量构成的矩阵W
-----------------------------------------------------------
pca = PCA(n_components=2) 指定主成分的个数，即降维后数据的维度
目标：
把鸢尾花数据4维降维成2维可视化出来
"""
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
 
data = load_iris()
y = data.target
print(y)
X = data.data
pca = PCA(n_components=2)
reduced_X = pca.fit_transform(X)
print(reduced_X)
# 按类别对降维后进行保存
red_x, red_y = [], []# 第一类数据点
blue_x, blue_y = [], []# 第二类数据点
green_x, green_y = [], []# 第三类数据点
# 按鸢尾花的类别把降维后的数据保存在不同的列表中
for i in range(len(reduced_X)):
    if y[i] == 0:
        red_x.append(reduced_X[i][0])
        red_y.append(reduced_X[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_X[i][0])
        blue_y.append(reduced_X[i][1])
    else:
        green_x.append(reduced_X[i][0])
        green_y.append(reduced_X[i][1])
 
plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()