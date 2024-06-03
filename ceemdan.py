#CEEMDAN分解
from copy import deepcopy

import numpy as np
from PyEMD import CEEMDAN
from PyEMD import Visualisation
from pylab import mpl
import matplotlib.pyplot as plt


def ceemdan_decompose(data, IImfs, plot):
    """
    :param plot: 是否画图
    :param data: 含噪信号
    :param IImfs: imf分量
    :return: imf分量
    """
    ceemdan = CEEMDAN()
    ceemdan.ceemdan(data)
    imfs, res = ceemdan.get_imfs_and_residue()
    mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    if(plot):
        plt.figure(figsize=(9,5))
        plt.plot(data,'r')
        vis = Visualisation()
        vis.plot_imfs(imfs = imfs,residue = res, include_residue = False,)

    for i in range(imfs.shape[0]):
        IImfs.append(imfs[i])
    return IImfs
import statistics
import math
def Pearson(X,Y):
    """
    :param X:
    :param Y:
    :return: 相似度
    """
    XY = X * Y
    EX = X.mean()
    EY = Y.mean()
    EX2 = (X ** 2).mean()
    EY2 = (Y ** 2).mean()
    EXY = XY.mean()
    numerator = EXY - EX * EY                                 # 分子
    denominator = math.sqrt(EX2 - EX ** 2) * math.sqrt(EY2 - EY ** 2) # 分母
    if denominator == 0:
        return 'NaN'
    rhoXY = numerator / denominator
    return rhoXY
def chooseImfs(originalSignal,IImfs):
    """
    选取imf分量
    :param originalSignal:原信号
    :param IImfs:imf
    :return:重构信号
    """
    IImfsNew = deepcopy(IImfs)
    Sim = []
    newSignal = []
    newSignal1 = []
    for imf in IImfsNew:
        sim = Pearson(imf,originalSignal)
        Sim.append(sim)
    print(Sim)
    for i in range(1, len(Sim) - 1):
        print(i)
        if Sim[i] < Sim[i - 1] and Sim[i] < Sim[i + 1]:
            minima = i
            print("第一个极小值点的下标为：", i)
            break
        if i == len(Sim) - 2:
            minima = i - 2
    for index in range(minima + 1, len(Sim)):
        if newSignal == []:
            if Sim[index] > 0.01:
                newSignal = IImfsNew[index]
                print('imf', index + 1)
        else:
            if Sim[index] > 0.01:
                newSignal += IImfsNew[index]
                print('imf', index + 1)
    #newSignal += IImfsNew[minima]
    newSignal1 = IImfsNew[-1]
    newSignal1 = newSignal1 + IImfsNew[-2] +IImfsNew[-3] +IImfsNew[-4]
    #return newSignal, Sim
    return newSignal1, Sim