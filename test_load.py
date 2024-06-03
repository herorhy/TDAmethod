from copy import deepcopy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
def normalization(X):
    # 计算每个特征的均值和标准差
    mu = np.mean(X)
    sigma = np.std(X)
    # 对X进行归一化
    X_norm = (X - mu) / sigma
    return X_norm
def loadcsv(name, index, plot):
    """
    :param plot:
    :param name: 沪深300数据集名称
    :param index: 用于选取成分股的下标
    :return: 成分股的名字以及数据
    """
    data = pd.read_csv("hs300cfg.csv")
    data.info()
    col_names = data.columns.tolist()
    nameOfcurve = col_names[index]
    print(nameOfcurve)
    y = data[nameOfcurve]
    x = data['日期']
    # 绘图
    if(plot):
        plt.figure(figsize=(9, 5))
        plt.plot(x, y, color='r')
        plt.title('中兴通讯')
        # plt.legend()图例
        # 显示中文标签
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        # 设置x轴间隔
        my_x_ticks = np.arange(0, 486, 141)  # 原始数据有x个点，故此处为设置从0开始，间隔为1
        plt.xticks(my_x_ticks)
        # 设置横纵坐标
        plt.xlabel('日期')
        plt.ylabel('收盘价(元)')
    # 显示图
    #plt.show()
    return nameOfcurve, normalization(deepcopy(y))