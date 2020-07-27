# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 21:44:50 2020

@author: Xu Bai

算法流程
+ 需要从特征文件和标签文件中将所有数据加载到内存中，由于存在缺失值，此步骤还需要进行
    简单的数据预处理
+ 创建对应的分类器，并使用训练数据进行训练
+ 利用测试集预测，计算准确率和召回率
可视化软件Weka
"""
import numpy as np
import pandas as pd

# 预处理模块
from sklearn.preprocessing import Imputer
# 自动生成训练集和测试集的模块
from sklearn.model_selection import train_test_split
# 预测结果评估函数
from sklearn.metrics import classification_report

# 同时导入3个分类器模块
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

def load_datasets(feature_paths, label_paths):
    '''读取特征文件列表和标签文件列表中的内容，归并后返回'''
    # 空的列表变量
    feature = np.ndarray(shape=(0, 41))
    label = np.ndarray(shape=(0,1))
    for file in feature_paths:
        # 使用逗号分隔符读取特征数据，将问号替换为缺失值，文件中不包含表头
        df = pd.read_table(file, delimiter=',', na_values='?', header=None)
        # 使用平均值补全缺失值，然后将数据进行补全
        imp = Imputer(missing_values='NaN', strategy='mean',axis=0)
        # 训练预处理器
        imp.fit(df)
        # 生成预处理结果
        df = imp.transform(df)
        # 将新读入的数据合并到特征集合中
        feature = np.concatenate((feature,df))
    for file in label_paths:
        df = pd.read_table(file, header=None)
        label = np.concatenate((label, df))
    # 将标签规整为一维向量
    label = np.ravel(label)
    return feature, label

if __name__ == "__main__":
    featurePaths = ['A/A.feature','B/B.feature','C/C.feature','D/D.feature','E/E.feature']
    labelPaths = ['A/A.label','B/B.label','C/C.label','D/D.label','E/E.label']
    # 前4个值作为训练集并作为参数传入load_dataset(),得到特则会那个
    x_train,y_train = load_datasets(featurePaths[:4],labelPaths[:4])
    x_test,y_test = load_datasets(featurePaths[4:],labelPaths[4:])
    # 使用全量数据作为训练集，借助train_test_split()的test_size=0将训练数据打乱
    x_train, x_, y_train, y_ = train_test_split(x_train, y_train, test_size = 0.0)
    
    print('Start training knn')
    knn = KNeighborsClassifier().fit(x_train, y_train)
    print('Training done')
    answer_knn = knn.predict(x_test)
    print('Prediction done')
     
    print('Start training DT')
    dt = DecisionTreeClassifier().fit(x_train, y_train)
    print('Training done')
    answer_dt = dt.predict(x_test)
    print('Prediction done')
     
    print('Start training Bayes')
    gnb = GaussianNB().fit(x_train, y_train)
    print('Training done')
    answer_gnb = gnb.predict(x_test)
    print('Prediction done')
    # 计算准确率与召回率
    print('\n\nThe classification report for knn:')
    print(classification_report(y_test, answer_knn))
    print('\n\nThe classification report for DT:')
    print(classification_report(y_test, answer_dt))
    print('\n\nThe classification report for Bayes:')
    print(classification_report(y_test, answer_gnb))