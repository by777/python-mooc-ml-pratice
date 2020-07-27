# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 21:48:52 2020

@author: Xu Bai
"""

import numpy as np
import sklearn.cluster as skc
from sklearn import metrics
import matplotlib.pyplot as plt
 
'''
DBSCAN密度聚类
-----------------------------
DBSCAN是一种基于密度的聚类算法，
+ 聚类的时候不需要预先指定簇的个数k
+ 最终的簇的个数也不确定
------------------------------
DBCSCAN把数据点分为三类：
+ 核心点：在半径Eps内含有超过MinPts数目的点
+ 边界点：在半径Eps内点的数量小于MinPts，但是落在核心点的邻域内
+ 噪音点：既不是核心点也不是边界点的点
------------------------------------
DBSCAN算法流程：
+ 将所有点标记为核心点、边界点或噪声点
+ 删除噪声点
+ 为距离在Eps之内的所有核心点之间赋予一条边
+ 每组联通的核心点形成一个簇
+ 将每个边界点指派到一个与之相关联的核心点的簇中（
    哪一个核心点的半径范围之内）
--------------------------------------
DBSCAN 主要参数：
    eps:两个样本被看作邻居节点的最大距离
    min_samples:簇的样本数
    metric:距离的计算方式
    
'''
# key是mac地址、value是对应mac地址的上网时长和开始上网时间
mac2id=dict()
onlinetimes=[]
f = open("../dataSet/TestData.txt", encoding='UTF-8')
'''
数据样式：
2c929293466b97a6014754607e457d68,
U201215025,
A417314EEA7B,                       ---------mac地址
10.12.49.26,
2014-07-20 22:44:18.540000000,      ------ 22:44:18 starttime
2014-07-20 23:10:16.540000000,
1558,                           -----------onlinetime
15,
本科生动态IP模版,
100元每半年,
internet
'''
for line in f:
    mac = line.split(',')[2]# mac地址
    onlinetime = int(line.split(',')[6])# 开始上网时间
    starttime = int(line.split(',')[4].split(' ')[1].split(':')[0])# 上网时长
    if mac not in mac2id:
        # key是mac地址，value是上网时长和开始上网时间
        mac2id[mac] = len(onlinetimes)
        onlinetimes.append((starttime,onlinetime))
    else:
        onlinetimes[mac2id[mac]] = [(starttime,onlinetime)]
# print(mac2id)
# print(onlinetimes)
real_X=np.array(onlinetimes).reshape((-1,2))
print("real_X:",real_X)
X = real_X[:,0:1]
print("X",X)
db = DBSCAN(eps=0.01,min_samples=20).fit(X)
# labels为数据的簇标签
labels = db.labels_
print("Labels:")
print(labels)
# 计算标签为-1噪声的比率
raito = len(labels[labels[:] == -1] / len(labels))
print('Noise raito:',format(raito, '.2%'))
# 计算簇的个数并进行打印
# set() 函数创建一个无序不重复元素集
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('Estimated number of clusters: %d' % n_clusters_)
# 评价聚类效果
print("Silhouette Coefficient: %0.3f"% metrics.silhouette_score(X, labels))
# 打印各簇标号以及各簇内数据
for i in range(n_clusters_):
    print("Cluster",i,":")
    print(list(X[labels == i].flatten()))
# 直方图
plt.hist(X,24)