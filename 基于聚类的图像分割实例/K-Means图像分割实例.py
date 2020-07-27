# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 23:39:11 2020

@author: Xu Bai

图像分割：
利用图像的灰度、颜色、纹理、形状等特征，把图像分成若干个互不重叠的区域，并使这些特征在
同一区域内呈现相似性，在不同的区域之间存在明显的差异性。然后就可以将分割的图像中具有独特
性质的区域提取出来用作不同的研究。

常用方法：
+ 阈值分割：对对象灰度值进行度量，设置不同类别的阈值
+ 边缘分割：对图像边缘进行检测，即图像中灰度值发生跳变的地方，则为一片区域的边缘
+ 直方图法：对图像的颜色建立直方图，而直方图的波峰波谷能够表示一块区域的颜色值的范围
+ 特定理论：基于聚类分析，小波变换等理论

输出：
同一聚类中的点使用相同颜色标记，不同聚类颜色标记不同
"""
import numpy as np
import PIL.Image as image
from sklearn.cluster import KMeans

def loadData(filePath):
    f = open(filePath,'rb')
    data = []
    img = image.open(f)
    m,n = img.size
    for i in range(m):
        for j in range(n):
            x,y,z =img.getpixel((i,j))
            data.append([x/256.0,y/256.0,z/256.0])
    f.close()
    # 以矩阵的形式返回data和图像大小
    return np.mat(data),m,n

imgData,row,col = loadData("bull.jpg")
label = KMeans(n_clusters=4).fit_predict(imgData)
label = label.reshape([row,col])
pic_new = image.new("L",(row,col))
for i in range(row):
    for j in range(col):
        pic_new.putpixel((i,j),int(256/(label[i][j]+1)))
pic_new.save("result.jpg","JPEG")