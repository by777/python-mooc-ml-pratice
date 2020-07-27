# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 22:24:54 2020

@author: Xu Bai
上证指数涨跌预测

实验目的：
根据给出时间前150天的历史数据，预测当天上证指数的涨跌
技术路线：
sklearn.svm.SVC
选取5列特征：
收盘价、最高价、最低价、开盘价、成交量
------------------------------
交叉验证思想
先将数据集D划分为k个大小相似的互斥子集，每个子集都尽可能保持数据分布的一致性，
即从D中通过分层采样得到。
然后，每次用k-1个子集的并集作为训练集，余下的那个子集作为测试集；
这样就可以获得k组训练/测试集，从而可以进行k次训练和测试，最终返回的是这个k个测试结果的均值
。通常把交叉验证称为‘K者交叉验证“，K最常用的取值是10
"""

import pandas as pd
import numpy as np
from sklearn import svm
# 在0.18以上的sklearn版本中，cross_validation sklearn.learning_curve
# sklearn.grid_search，都在model_selection下面
from sklearn.model_selection import train_test_split
data = pd.read_csv('000777.csv',encoding='gbk',parse_dates=[0],index_col=0)
# 按照事件升序
data.sort_index(0,ascending=True,inplace=True)

print(data)
# 选取150天
dayfeature = 150
# 选取的特征
featurenum = 5 * dayfeature
# 150天的5个特征值
# data.shape[0] - dayfeature意思是因为我们要用150天数据做训练，
# 对于条目为200条的数据，只有50条数据是前150天的数据来训练的，所以训练集大小就是
# 200 - 150 ,对于每一条数据，它的特征是前150天的所有特征数据，即150 * 5 ， +1是
# 将当天开盘价引入作为一条特征数据
x = np.zeros((data.shape[0]-dayfeature,featurenum+1))
# 记录涨或跌
y = np.zeros((data.shape[0]-dayfeature))

for i in range(0,data.shape[0]-dayfeature):
    x[i,0:featurenum]=np.array(data[i:i+dayfeature] \
          [[u'收盘价',u'最高价',u'最低价',u'开盘价',u'成交量']]).reshape((1,featurenum))
    x[i,featurenum]=data.ix[i+dayfeature][u'开盘价']
for i in range(0,data.shape[0]-dayfeature):
    if data.ix[i+dayfeature][u'收盘价']>=data.ix[i+dayfeature][u'开盘价']:
        y[i]=1
    else:
        y[i]=0 
# 如果当天收盘价高于开盘价 ， y[i]=1代表涨否则0跌
# 默认rbf，linear,poly,sigmoid
clf=svm.SVC(kernel='rbf')
result = []
for i in range(5):
    # 切分训练集测试集
    x_train, x_test, y_train, y_test = \
                train_test_split(x, y, test_size = 0.2)
    clf.fit(x_train, y_train)
    result.append(np.mean(y_test == clf.predict(x_test)))
print("svm classifier accuacy:")
print(result)