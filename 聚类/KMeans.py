# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 22:33:14 2020

@author: Xu Bai
"""

import numpy as np
from sklearn.cluster import KMeans
 
 
def loadData(filePath):
    fr = open(filePath,'r+')
    lines = fr.readlines()
    retData = []
    retCityName = []
    for line in lines:
        items = line.strip().split(",")
        retCityName.append(items[0])
        for i in range(1,len(items)):
            retData.append([float(items[i])])
        #等价于retData.append([float(items[i]) for i in range(1,len(items))])
    
    #print(retData)
    return retData,retCityName
 
     
if __name__ == '__main__':
    data,cityName = loadData('../dataSet/city.txt')
    km = KMeans(n_clusters=4)
    label = km.fit_predict(data)
    expenses = np.sum(km.cluster_centers_,axis=1)
    #expense是聚类中心的数值加和，也就是平均消费水平
    #print(expenses)
    CityCluster = [[],[],[],[]]
    for i in range(len(cityName)):
        CityCluster[label[i]].append(cityName[i])
    for i in range(len(CityCluster)):
        print("Expenses:%.2f" % expenses[i])
        print(CityCluster[i])