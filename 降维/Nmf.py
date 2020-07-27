# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 22:22:11 2020

@author: Xu Bai
---------------
NMF（Non-negative Matrix Factorization）非负矩阵分解
是在矩阵中所有元素均为非负数约束条件下的矩阵分解方法

基本思想：
给定一个非负矩阵V，NMF能够找到一个非负矩阵W和一个非负矩阵H，使得矩阵W和H的乘积
近似等于矩阵中V的值 V_nm = W_nk * H_km

W矩阵：基础图现象矩阵，相当于从源实矩阵V中抽取出来的特征
H矩阵：系数矩阵

NMF能够广泛应用于图像分析、文本挖掘和语音处理等领域

NMF目标：
最小化W矩阵与H矩阵乘积与原始矩阵之间的差别

-------------------------------------------------
Olivetti人脸数据集提供了400张64 * 64的人脸图像
k=6 设置特征的数目
"""
# 随机化种子
from numpy.random import RandomState
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn import decomposition

dataset = fetch_olivetti_faces(shuffle=True,random_state=RandomState(0))
faces = dataset.data

# 图像展示时的排列情况
n_row,n_col = 2,3
# 提取的特征的数目
n_components = n_row * n_col
# 人脸数据图片的大小
image_shape = (64,64)

# 图片的展示方式
def plot_gallery(title,images,n_col=n_col,n_row=n_row):
    # 创建图片指定大小
    plt.figure(figsize=(2. * n_col,2.26 * n_row))
    # 标题及字号大小
    plt.suptitle(title,size=16)
    for i,comp in enumerate(images):
        # 选择画制的子图
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        # 归一化并灰度
        '''
        imshow()接收一张图像，只是画出该图，并不会立刻显示出来。
        imshow后还可以进行其他draw操作，比如scatter散点等。        
        所有画完后使用plt.show()才能进行结果的显示。
        '''
        plt.imshow(comp.reshape(image_shape),cmap=plt.cm.gray,interpolation='nearest',
                 vmin=-vmax,vmax=vmax,)
        # 去除子图的坐标轴标签
        plt.xticks(())
        plt.yticks(())
        # 对子图位置及间隔调整
        plt.subplots_adjust(0.01,0.05,0.99,0.93,0.04,0.)
     
plot_gallery("First centered Olivetti faces", faces[:n_components])

estimators = [
        # PCA用于对比
    ('Eigenfaces - PCA using randomized SVD',
         decomposition.PCA(n_components=6,whiten=True)), 
    ('Non-negative components - NMF',
         decomposition.NMF(n_components=6, init='nndsvda', tol=5e-3))
]

for name, estimator in estimators:
    print("Extracting the top %d %s..." % (n_components, name))
    print(faces.shape)
    estimator.fit(faces)
    components_ = estimator.components_
    plot_gallery(name, components_[:n_components])
 
plt.show()

 