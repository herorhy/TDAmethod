#得到参数后进行相空间重构
import numpy as np


def embed_vectors_1d(signal, lag, embed):
    """Embeds vectors from a one dimensional time series in m-dimensional
    space.

    Parameters
    ----------
    X : 1d array
        Training or testing set.
    lag : int
        Lag value as calculated from the first minimum of the mutual info.
    embed : int
        Embedding dimension. How many lag values to take.
    predict : int
        Distance to forecast (see example).

    Returns
    -------
    features : 2d array
        Contains all of the embedded vectors. Shape (num_vectors,embed).

    Example
    -------
    >>> X = [0,1,2,3,4,5,6,7,8,9,10]
    em = 3
    lag = 2
    predict=3

    >>> embed_vectors_1d
    features = [[0,2,4], [1,3,5], [2,4,6], [3,5,7]]
    """

    tsize = signal.shape[0]
    t_iter = tsize - (lag * (embed - 1))

    features = np.zeros((t_iter, embed))

    for ii in range(t_iter):

        end_val = ii + lag * (embed - 1) + 1

        part = signal[ii : end_val]

        features[ii,:] = part[::lag]

    return features
#零-均值归一化函数
def normalization(X):
    # 计算每个特征的均值和标准差
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    # 对X进行归一化
    X_norm = (X - mu) / sigma
    return X_norm