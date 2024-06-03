import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics

import ceemdan
import test_load


def normalization(X):
    # 计算每个特征的均值和标准差
    mu = np.mean(X)
    sigma = np.std(X)
    # 对X进行归一化
    X_norm = (X - mu) / sigma
    return X_norm

test_data=pd.read_csv('final.csv',encoding='GBK')
data = np.array(test_data[['1_max','2_max','1_avg','2_avg','瓶颈距离','一维持续性景观的L1范数','一维持续性景观的L2范数 ','持续熵']])
"""
#下面是实验，只加AE
x=[]
fliename = "hs300cfg.csv"
for i in range(1,289):
    nameOfcurve, originalSignal = test_load.loadcsv(fliename,i, 0)
    x.append(originalSignal)
data=np.array(x)
"""
for i in data:
    for j in range(0,8):
        if i[j]== 0:
            i[j]=0.001

data = np.log(data)

#print(data)
#data = normalization1(data)
#print(data)

#-------------------------------------------------------------------------------------------------------------------
k = 4
# 调用k-means算法
model = KMeans(n_clusters=k, init= 'k-means++',random_state=28)# n_clusters是聚类数，即k；init是初始化方法，可以是'k-means ++'，'random'或者自定义，default=’k-means++’
model.fit(data)
print("聚类中心")
#print(model.cluster_centers_)
print("结果")
print(model.labels_)
labels = model.labels_
#-------------------------------------------------------------------------------------------------------------------

def curveShow(fliename,labels,index):
    num = 0
    Y=[]
    name = []
    for i in labels:
        num += 1
        if i == index:
            nameOfcurve, originalSignal = test_load.loadcsv(fliename, num, 0)
            originalSignal = originalSignal.to_numpy()
            #IImfs = []
            #IImfs = ceemdan.ceemdan_decompose(originalSignal, IImfs, 0)
            #IImfs = np.array(IImfs)
            # 得到新信号和相似度列表
            #newSignal, Sim = ceemdan.chooseImfs(originalSignal, IImfs)
            #Y.append(normalization(newSignal))
            Y.append(originalSignal)
            name.append(nameOfcurve)

            #Y.append(newSignal)
    fig, ax = plt.subplots()
    for curve in Y:
        y = np.array(curve)
        x = np.arange(len(curve))
        ax.plot(x, y, color='blue', linestyle='-')
        #plt.plot(x, y, color='blue', linestyle='-')
        #plt.show()
    Yaverage = np.average(Y,axis=0)
    ax.plot(x, Yaverage, color='red', linestyle='-')
    return name
    #plt.show()
filename = "hs300cfg.csv"
for i in range(0, k):
    name = curveShow(filename, model.labels_, i)
    print(name)
    plt.show()
#curveShow(filename, model.labels_, 0)
#plt.show()
ss = metrics.silhouette_score(data, labels)
ch = metrics.calinski_harabasz_score(data, labels)
print("silhouette_score:", ss)
print("calinski-harabaz_score:", ch)
