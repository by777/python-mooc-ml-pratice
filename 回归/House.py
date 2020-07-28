# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 22:50:27 2020

@author: Xu Bai
线性回归（Liner Regression）
利用数理统计中回归分析，来确定两种或两种以上变量相互依赖的定量关系的一种统计分析方法。
线性回归利用线性回归方程的最小平方函数对一个或多个自变量之间的关系进行建模。这种函数
是一个或多个称为回归系数的模型参数的线性组合，
只有一个自变量的情况叫做简单回归，大于一个自变量情况的叫多元回归。

"""
from matplotlib import pyplot as plt
from sklearn import linear_model
import numpy as np
# 房屋尺寸
datasets_X = []
# 房屋价格
datasets_Y = []

fr = open('prices.txt','r')
lines = fr.readlines()
fr.close()

for line in lines:
    # strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
    items = line.strip().split(',')
    datasets_X.append(int(items[0]))
    datasets_Y.append(int(items[1]))
length = len(datasets_X)
datasets_X = np.array(datasets_X).reshape([length,1])
datasets_Y = np.array(datasets_Y)

minX = min(datasets_X)
maxX = max(datasets_X)
# 取范围简历等差数列方便画图
X = np.arange(minX,maxX).reshape([-1,1])

# 参数：fit_intercept：是否计算截距
# normalize=False:归一化
linear = linear_model.LinearRegression()

linear.fit(datasets_X,datasets_Y)
# 查看回归方程系数：
print("Coefficients:",linear.coef_)
# 查看截距
print("intercept:",linear.intercept_)
# 散点图
plt.scatter(datasets_X,datasets_Y,color='red')
# 直线
plt.plot(X,linear.predict(X),color='blue')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()