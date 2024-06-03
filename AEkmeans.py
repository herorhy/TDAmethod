import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from sklearn import metrics
import test_load

#加载数据
data = pd.read_csv('jiangwei.csv', encoding='GBK')
x = []
for i in range(1,289):
    x.append(data[str(i)])
x=np.array(x)

k = 4
# 调用k-means算法
model = KMeans(n_clusters=k, init= 'k-means++',random_state=28)# n_clusters是聚类数，即k；init是初始化方法，可以是'k-means ++'，'random'或者自定义，default=’k-means++’
model.fit(x)
print("聚类中心")
#print(model.cluster_centers_)
print("结果")
print(model.labels_)
labels = model.labels_
def normalization(X):
    # 计算每个特征的均值和标准差
    mu = np.mean(X)
    sigma = np.std(X)
    # 对X进行归一化
    X_norm = (X - mu) / sigma
    return X_norm

def curveShow(fliename,labels,index):
    num = 0
    Y=[]
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
    fig, ax = plt.subplots()
    for curve in Y:
        y = np.array(curve)
        x = np.arange(len(curve))
        ax.plot(x, y, color='blue', linestyle='-')
        #plt.plot(x, y, color='blue', linestyle='-')
        #plt.show()
    Yaverage = np.average(Y, axis=0)
    ax.plot(x, Yaverage, color='red', linestyle='-')
    #plt.show()
filename = "hs300cfg.csv"
for i in range(0, k):
    curveShow(filename, model.labels_, i)
#curveShow(filename, model.labels_, 0)
plt.show()

ss = metrics.silhouette_score(x, labels)
ch = metrics.calinski_harabasz_score(x,labels)
print("silhouette_score:", ss)
print("calinski-harabaz_score:", ch)


