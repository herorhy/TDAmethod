# 互信息法确定相空间重构的时间延迟
import numpy as np


def pmf_get(datalist, n):
    # histogram 直方图
    [occur, bin] = np.histogram(datalist, bins=n)
    pmf = occur[0: n] / len(datalist)
    # bin得到每个分箱的均值
    # pmf （probability mass function，离散型）, 也叫 pdf (Probability density function，连续型)，概率密度函数
    return pmf, bin


# 联合概率
def jointProb_get(jointdata, n):
    # histogramdd 多维直方图
    jointProbs, edges = np.histogramdd(jointdata, bins=[n, n])
    jointProbs /= jointProbs.sum()
    return jointProbs, edges


# 求信息熵
def info_entropy_single(data_list, bin, n, pmf):
    H_s = 0
    for data in data_list:
        for j in range(n):
            if bin[j] <= data < bin[j + 1]:
                H_s += (-pmf[j] * np.log2(pmf[j]))
    return H_s


def mutual_information(data_list, taustep=1, n=50):
    """
    :param data_list: list 时间序列数据
    :param taustep: int tau间隔
    :param n: int 分箱数
    :return: int 最优延迟数
    """
    I_sq_list = []
    tau = 0  # 时间延迟
    while 1:
        # print(tau)
        # list0:假设5个数据，得到等差数组，也就是下标 0，1，2，3，4    0，1，2，3 ...
        # list1:向后延迟tau个时间，得到等差数组，也就是下标 0，1，2，3，4    1，2，3，4 ...
        select_list0 = np.arange(0, len(data_list) - tau)
        select_list1 = select_list0 + tau
        # x0,x1为创建上述下标之后的源数据中对应下标的值
        x0 = [data_list[i] for i in select_list0]
        x1 = [data_list[i] for i in select_list1]
        # 联合数据
        jointdata = np.array([[x0[i], x1[i]] for i in range(len(x0))])
        # 得到P(s) P(Q)
        pmf0, bin0 = pmf_get(x0, n)
        pmf1, bin1 = pmf_get(x1, n)
        # 得到P(s,q)
        jointProbs, edges = jointProb_get(jointdata, n)
        I_sq = 0
        for data in jointdata:
            for i in range(n):
                for j in range(n):
                    if edges[0][i] <= data[0] < edges[0][i + 1] and edges[1][j] <= data[1] < edges[1][j + 1]:
                        # 求互信息
                        I_sq = I_sq + (jointProbs[i, j] * np.log2((jointProbs[i, j]) / (pmf0[i] * pmf1[j])))
        I_sq_list.append(I_sq)
        if len(I_sq_list) > 2:
            # 取第一个局部极小值
            if I_sq_list[int(tau / taustep) - 1] - I_sq_list[int(tau / taustep - 1) - 1] > 0 and I_sq_list[
                int(tau / taustep - 1) - 1] - I_sq_list[int(tau / taustep - 2) - 1] < 0:
                tau_selected = tau - 1
                print("最优时间延迟", tau_selected)
                break
        tau += taustep
    return I_sq_list, tau_selected
#计算GP算法的嵌入维数(假近邻算法)
def get_m(imf, tau):
    N = len(imf)
    m_max = 10
    P_m_all = []  # m_max-1个假近邻点百分率
    for m in range(1, m_max + 1):
        i_num = N - (m - 1) * tau
        kj_m = np.zeros((i_num, m))  # m维重构相空间
        for i in range(i_num):
            for j in range(m):
                kj_m[i][j] = imf[i + j * tau]
        if (m > 1):
            index = np.argsort(Dist_m)
            a_m = 0  # 最近邻点数
            for i in range(i_num):
                temp = 0
                for h in range(i_num):
                    temp = index[i][h]
                    if (Dist_m[i][temp] > 0):
                        break
                D = np.linalg.norm(kj_m[i] - kj_m[temp])
                D = np.sqrt((D * D) / (Dist_m[i][temp] * Dist_m[i][temp]) - 1)
                if (D > 10):
                    a_m += 1
            P_m_all.append(a_m / i_num)
        i_num_m = i_num - tau
        Dist_m = np.zeros((i_num_m, i_num_m))  # 两向量之间的距离
        for i in range(i_num_m):
            for k in range(i_num_m):
                Dist_m[i][k] = np.linalg.norm(kj_m[i] - kj_m[k])
    P_m_all = np.array(P_m_all)
    m_all = np.arange(1, m_max)
    #选取最优嵌入维数m
    P_m_all_last = 0 #上一个假近邻率
    int_m = 0 #计数器
    m_best = 0 #最优嵌入维数
    for m in P_m_all:
        if  np.abs(P_m_all_last - m) <= 0.05:
            m_best = int_m
            break
        P_m_all_last = m
        int_m = int_m + 1
    if(m_best<3):
        m_best = 3
    return m_all, P_m_all,m_best