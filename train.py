'''
入侵检测的方法从根本上讲就是设计一个分类器，能将数据流中的正常与异常数据区分出来，从而实现对攻击行为的报警。
'''
import im
import os
import sys
import time
import csv
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

#随机森林
def RandomForest(sr):
   datas = pd.read_csv(sr,header=None,delimiter=',')
   # print(datas)
   dataset = np.array(datas)
   # print(dataset)
   print("数据集shape: ", dataset.shape)
   X = dataset[:,0:41]
   Y = dataset[:,41]
   X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1) # 划分训练集和测试集
   print("训练集个数：", X_train.shape[0])
   print("测试集个数：", X_test.shape[0])
   rf = RandomForestClassifier()
   # 拟合模型
   rf.fit(X_train, Y_train.astype(int))
   # print(rf)
   expected = Y
   predicted = rf.predict(X)
   # summarize the fit of the model
   # print(metrics.classification_report(expected, predicted))
   # print(metrics.confusion_matrix(expected, predicted))
   cross_val_score(rf, X, Y.astype(int), scoring=None, cv=None, n_jobs=1)



#KNN k-最近邻
def KNN(sr):
    datas = pd.read_csv(sr, header=None, delimiter=',')
    # print(datas)
    dataset = np.array(datas)
    # print(dataset)
    print('KNN:')
    print("数据集shape: ", dataset.shape)
    X = dataset[:, 0:41]
    Y = dataset[:, 41]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)  # 划分训练集和测试集
    print("训练集个数：", X_train.shape[0])
    print("测试集个数：", X_test.shape[0])
    knn = KNeighborsClassifier()
    # 拟合模型
    knn.fit(X_train, Y_train.astype(int))
    # print(knn)
    # make predictions
    expected = Y_train.astype(int)
    predicted = knn.predict(X_train)
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))


#SVM支持向量机
def SVM(sr):
    datas = pd.read_csv(sr, header=None, delimiter=',')
    # print(datas)
    dataset = np.array(datas)
    # print(dataset)
    print('SVM:')
    print("数据集shape: ", dataset.shape)
    X = dataset[:, 0:41]
    Y = dataset[:, 41]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)  # 划分训练集和测试集
    print("训练集个数：", X_train.shape[0])
    print("测试集个数：", X_test.shape[0])
    svc = SVC()
    # 拟合模型
    svc.fit(X_train, Y_train.astype(int))
    # print(knn)
    # make predictions
    expected = Y_train.astype(int)
    predicted = svc.predict(X_train)
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))

#朴素贝叶斯
def GNB(sr):
    datas = pd.read_csv(sr, header=None, delimiter=',')
    # print(datas)
    dataset = np.array(datas)
    # print(dataset)
    print('朴素贝叶斯:')
    print("数据集shape: ", dataset.shape)
    X = dataset[:, 0:41]
    Y = dataset[:, 41]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)  # 划分训练集和测试集
    print("训练集个数：", X_train.shape[0])
    print("测试集个数：", X_test.shape[0])
    gnb = GaussianNB()
    # 拟合模型
    gnb.fit(X_train, Y_train.astype(int))
    # print(knn)
    # make predictions
    expected = Y_train.astype(int)
    predicted = gnb.predict(X_train)
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))



if __name__ == '__main__':
    pth = './trained_data2.csv'
    SVM(pth)
    GNB(pth)
    # KNN(pth)
    # RandomForest(pth)
