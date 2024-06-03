#库
import gudhi
import matplotlib.pyplot as plt
import numpy as np
import csv
#引用文件
import ceemdan
import parameter_selector
import phase_space_reconstruction
import Witness
import features_calculate
import test_load
#test of change
def TDAmethod(filename, index, finished, debugmode):
    """
    :param filename: 沪深300数据集名称
    :param index: 选择成分股，1-300
    :param finsished: 是否处理完成
    :param debugmode: 调试模式开关，当调试模式开启时，会有中间过程图
    :return:
    """
    if (finished):
        return
    # ___________________________________文件读取________________________________________________________________
    # 实际数据
    indexOfcurve = index
    nameOfcurve, originalSignal = test_load.loadcsv(filename, indexOfcurve, 1)
    if (debugmode):
        plt.show()
    print(nameOfcurve, "_____文件读取开始_____")

    print(nameOfcurve, "_____文件读取完成_____")
    # ___________________________________IMF分解________________________________________________________________
    # tips：记得设置全局变量 IImfs=[]
    print(nameOfcurve, "_____imf分解开始_____")
    originalSignal = originalSignal.to_numpy()
    IImfs = []
    IImfs = ceemdan.ceemdan_decompose(originalSignal, IImfs, 1)
    IImfs = np.array(IImfs)
    # 得到新信号和相似度列表
    newSignal, Sim = ceemdan.chooseImfs(originalSignal, IImfs)
    # 绘图


    plt.figure(figsize=(9, 5))
    plt.plot(Sim)
    plt.xlabel("模态函数")
    plt.ylabel("相似度")
    plt.scatter(3, Sim[3], c='r', marker='s')
    #plt.plot(Sim)
    if(debugmode):
        plt.show()

    plt.figure(figsize=(9, 5))
    plt.plot(originalSignal)
    plt.figure(figsize=(9, 5))
    plt.plot(newSignal, 'r')
    plt.xlabel('日期')
    plt.ylabel('收盘价(元)')
    plt.title("包钢股份")
    if(debugmode):
        plt.show()
    print(nameOfcurve, "_____imf重构完成_____")

    # ____________________________________确定时间延迟_______________________________________________________________
    print(nameOfcurve, "_____时间延迟计算开始_____")
    I_sq_list, tau_selected = parameter_selector.mutual_information(newSignal)
    t_all = np.arange(1, tau_selected + 3)
    plt.figure(figsize=(9, 5))
    plt.xlabel('时间tau')
    plt.ylabel('延迟互信息')
    plt.scatter(tau_selected, I_sq_list[tau_selected - 1], c='r', marker='s')
    plt.plot(t_all, I_sq_list, c='r')
    print(nameOfcurve, "_____时间延迟计算结束_____")
    # ____________________________________确定嵌入维度_______________________________________________________________
    print(nameOfcurve, "_____嵌入维度计算开始_____")
    m, P_m, m_best = parameter_selector.get_m(newSignal, tau_selected)
    plt.figure(figsize=(9, 5))
    plt.xlabel('嵌入维数m')
    plt.ylabel('假近邻率')
    plt.scatter(m_best, P_m[m_best - 1], c='r', marker='s')
    plt.plot(m, P_m, c='r')
    if (debugmode):
        plt.show()
    print("最优嵌入维度", m_best)
    print(nameOfcurve, "_____嵌入维度计算结束_____")
    # ____________________________________相空间重构_______________________________________________________________
    print(nameOfcurve, "_____相空间重构开始_____")
    features = phase_space_reconstruction.embed_vectors_1d(newSignal, tau_selected, m_best)
    plt.plot(features)
    plt.show()

    ax = plt.axes(projection="3d")
    ax.set_xlabel('x(t)')
    ax.set_ylabel('x(t-tau)')
    ax.set_zlabel('x(t-2*tau)')
    ax.plot(features[:, 0], features[:, 1], features[:, 2], color="red")
    if (debugmode):
        plt.show()
    print(nameOfcurve, "_____相空间重构结束_____")
    # ____________________________________持续同调_______________________________________________________________
    print(nameOfcurve, "_____持续同调开始_____")
    # max_min采样
    landmarks = Witness.max_min_sampling(features,35)
    bx = plt.axes(projection="3d")
    bx.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], color="red")
    bx.set_xlabel('x(t)')
    bx.set_ylabel('x(t-tau)')
    bx.set_zlabel('x(t-2*tau)')
    if (debugmode):
        plt.show()
    # 选择alpha参数
    max_alpha_square = Witness.select_max_alpha_square(features)
    # 构建witness复形
    witness = gudhi.EuclideanWitnessComplex(landmarks, features)
    # 构建单纯形树
    simplex_tree = witness.create_simplex_tree(max_alpha_square)
    # 计算单纯复形的维数.
    print('-----complex dimension:----', simplex_tree.dimension())
    # 计算整个复形的所有单形个数
    print('-----complex dimension:----', simplex_tree.num_simplices())
    # 计算构建单纯复形时的顶点数。
    print('-----complex dimension:----', simplex_tree.num_vertices())
    diag = simplex_tree.persistence(homology_coeff_field=2, min_persistence=0)
    print("diag", diag)
    font_stlyle = {
        #'family': 'Times New Roman',
        'family': 'Microsoft YaHei',
        'weight': 'normal',
        'size': 18,
        # 'style': 'italic'
    }
    # 持续性图
    gudhi.plot_persistence_diagram(diag, inf_delta=0.3)
    #plt.title("Persistence diagram of a Witness complex", font_stlyle)
    plt.title("持续性图", font_stlyle)
    plt.xlabel('产生', font_stlyle)
    plt.ylabel('消亡', font_stlyle)
    if (debugmode):
        plt.show()
    print(nameOfcurve, "_____持续同调结束_____")
    # ____________________________________特征提取与处理_______________________________________________________________
    print(nameOfcurve, "_____特征处理开始_____")
    features_single = []
    # 得到单独特征
    features_single = features_calculate.calculate_features(diag, max_alpha_square)
    # 将前两个特征拆分，并把其他特征加入
    #print(features_single)
    features_single_normal = [[] for j in range(300)]
    features_all = []
    # 先加入序号和名称
    features_single_normal[index].append(indexOfcurve)
    features_single_normal[index].append(nameOfcurve)
    features_single_normal[index].append(features_single[0][1])
    features_single_normal[index].append(features_single[0][2])
    features_single_normal[index].append(features_single[1][1])
    features_single_normal[index].append(features_single[1][2])
    """
    for i in features_single[0]:
        features_single_normal.append(i)
    for i in features_single[1]:
        features_single_normal.append(i)
    """
    for i in range(2, 6):
        features_single_normal[index].append(features_single[i])
    print( features_single_normal[index])
    features_all.append(features_single_normal[index])
    print("开始写入特征")
    filename = 'features_single_normal.csv'
    #filename = '2.csv'
    # 判断文件是否存在，如果不存在，则先写入表头
    with open(filename, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if csv_file.tell() == 0:  # 如果文件位置指针在初始位置（即文件为空）
            writer.writerow(['序号', '名称', '1_max', '2_max', '1_avg', '2_avg', '瓶颈距离',
                             '一维持续性景观的L1范数', '一维持续性景观的L2范数 ', '持续熵'])
        # 写入数据
        writer.writerow(features_single_normal[index])
        #del features_single_normal
    print(nameOfcurve, "_____特征处理结束_____")
    #print(features_single_normal)


if __name__ == '__main__':
    debugmode = 1
    if(debugmode):
        index = 118
        TDAmethod("hs300cfg.csv", index, 0, 1)
    else:
        for i in range(242, 290):
            print(i)
            TDAmethod("hs300cfg.csv", i, 0, 0)

