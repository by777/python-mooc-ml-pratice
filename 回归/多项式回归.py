# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 22:08:38 2020

@author: Xu Bai
多项式回归 + 
岭回归 + 

房价与房屋尺寸的关系
----------------------------------------
多项式回归（Polynomial Regression）是研究一个因变量与一个或多个自变量间多项式的回归
分析方法。

在一元回归分析中，如果因变量y与自变量x的关系为非线性的，但是又找不到适合的函数曲线来
拟合，则可以采用一元多项式回归

多项式回归的最大优点就是可以通过增加x的高次项对实测点逼近，直至满意

事实上，多项式回归可以处理相当一类非线性问题，他在回归分析中占有非常重要的地位，因为
任意函数都可以分段用多项式来逼近。

在线性回归中，是运用直线来拟合数据的输入与输出之间的线性关系。不同于线性回归，
多项式回归是使用曲线拟合数据的输入与输出关系的映射关系。

sklearn中的多项式回归，实际上是先将变量X处理成多项式特征，然后使用线性模型学习多项式
特征的参数，以达到多项式回归的目的。

eg。
X = [x1, x2]
1. 使用PolynomialFeatures构造X的二次多项式特征X_Poly
    X_Poly = [x1, x2, x1x2, x1^2, x2^2]
2. 使用linear_model学习X_Poly和y之间的映射关系，即参数
    w1x1 + w2x2 + w3x1x2 + w4x1^2 + w5x2^2 = y

# 建立datasets_X的二次多项式特征X_poly。然后创建线性回归，使用线性模型学习X_poly和
# datasets_Y之间的映射关系（即参数）
poly_reg = PolynomialFeatures(degree = 2) 
X_poly = poly_reg.fit_transform(datasets_X)
lin_reg_2 = linear_model.LinearRegression()
lin_reg_2.fit(X_poly, datasets_Y)
---------------------------------
岭回归(ridge regression)
一种专用于共线性数据分析的有偏估计回归方法。
是一种改良的最小二乘估计法，对某些数据的拟合要强于最小二乘法。

sklearn.linear_model.Ridge(alpha=正则化因子,fit_intercept:是否计算截距,solver：
                           计算参数的方法'auto \ svd \ sag等')
数据加载：
data = np.genfromtxt('data.txt')
plt.plot(data[:,4])

数据处理：
X = data[:,:4] # 0~3维 属性
y = data[:,4] # 4维 车流量
ploy = PloynomialFeatures(6) # 6次多项式
X = ploy.fit_transform(X) # X 为创建的多项式

划分训练集和测试集
train_set_X,test_set_X,train_set_y,test_set_y = \
    cross_validation.train_test_split(X,y,test_size=0.3,random_state=0)

创建分类器进行训练
clf = Ridge(alpha=1.0,fit_intercept=True)
# 训练
clf.fit(train_set_X,train_set_y)
clf.score(test_set_X,test_set_Y)

岭回归的优化目标：
argmin||Xw - y ||^2 + α||w||^2

"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
 
 
# 读取数据集
datasets_X = []
datasets_Y = []
fr = open('prices.txt','r')
lines = fr.readlines()
for line in lines:
    items = line.strip().split(',')
    datasets_X.append(int(items[0]))
    datasets_Y.append(int(items[1]))
 
length = len(datasets_X)
datasets_X = np.array(datasets_X).reshape([length,1])
datasets_Y = np.array(datasets_Y)
 
minX = min(datasets_X)
maxX = max(datasets_X)
X = np.arange(minX,maxX).reshape([-1,1])
 
 
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(datasets_X)
lin_reg_2 = linear_model.LinearRegression()
lin_reg_2.fit(X_poly, datasets_Y)
 
# 图像中显示
plt.scatter(datasets_X, datasets_Y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()