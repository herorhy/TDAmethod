import gudhi
import numpy as np
from gudhi.representations import Landscape
#将持续同调按维度分组
def calculate_features(diag,max_alpha_square):
    """
    计算特征
    :param diag: 持续性图
    :param max_alpha_square: 用于控制分量个数
    :return: 得到特征用于保存
    """
    features_single = []  # 特征
    diag0, diag1, diag2 = [],[],[]
    diagAll = []
    for i in diag:
        if(i[0] == 2):
            #若为无穷，则替换为max_alpha_square
            if(np.isinf(i[1][1])):
                diag2.append([i[1][0], 10000])
            else:
                diag2.append(i[1])
        if(i[0] == 1):
            if(np.isinf(i[1][1])):
                diag1.append([i[1][0], 10000])
            else:
                diag1.append(i[1])
        if(i[0] == 0):
            if(np.isinf(i[1][1])):
                diag0.append([i[1][0], 10000])
            else:
                diag0.append(i[1])
    #计算不同维度下洞的持续时间
    pers0, pers1, pers2 = [],[],[]
    if (diag2 == []):
        pers2.append(0)
        if(diag1 == []):
            pers1.append(0)
            for i in diag0:
                pers0.append(i[1] - i[0])
        else:
            for i in diag1:
                pers1.append(i[1] - i[0])
            for i in diag0:
                pers0.append(i[1] - i[0])
    else:
        for i in diag2:
            pers2.append(i[1]-i[0])
        if (diag1 == []):
            pers1.append(0)
            for i in diag0:
                pers0.append(i[1] - i[0])
        else:
            for i in diag1:
                pers1.append(i[1] - i[0])
            for i in diag0:
                pers0.append(i[1] - i[0])
    #print(pers2,pers1,pers0)
    #print(np.max(pers2))
    #计算不同维度下洞的最长持续时间
    persistence_max012=[]
    persistence_max012.append(np.max(pers0))
    persistence_max012.append(np.max(pers1))
    persistence_max012.append(np.max(pers2))
    features_single.append(persistence_max012)
    print("0,1,2维下洞的最长持续时间为",persistence_max012)
    #计算不同维度下洞的平均持续时间
    persistence_avg012=[]
    persistence_avg012.append(np.mean(pers0))
    persistence_avg012.append(np.mean(pers1))
    persistence_avg012.append(np.mean(pers2))
    features_single.append(persistence_avg012)
    print("0,1,2维下洞的平均持续时间为",persistence_avg012)
    #计算瓶颈距离
    diagAll = diag1 + diag2 #diagAll为所有维度特征的集合，去除了维度
    diagAll = np.array(diagAll)
    #print("diagall", diagAll)
    diagempty=[]#创建只包含对角线的持续性图用于计算瓶颈距离
    bottleneck_distance = gudhi.bottleneck_distance(diagAll, diagempty)
    features_single.append(bottleneck_distance)
    print("瓶颈距离",bottleneck_distance)
    #计算持续性景观
    diags = [diagAll]
    if(diag1 == [] and diag2 == []):
        L = 0
        print("ssssss")
    else:
        L = Landscape(num_landscapes=2, resolution=10).fit_transform(diags)
    L1 = np.linalg.norm(np.ravel(L), ord=1)
    L2 = np.linalg.norm(np.ravel(L), ord=2)
    features_single.append(L1)
    features_single.append(L2)
    print("一维持续性景观的L1范数",L1)
    print("一维持续性景观的L2范数",L2)
    #计算持续熵
    if (diag1 == [] and diag2 == []):
        features_single.append(-1)
    else:
        entropy = gudhi.representations.vector_methods.Entropy()(diagAll)
        features_single.append(entropy[0])
        print("持续熵", entropy[0])
    return features_single