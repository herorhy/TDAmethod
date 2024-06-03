import numpy as np
import pandas as pd
from datetime import date
import numpy.random as npr
import matplotlib.pyplot as plt
from pylab import mpl
import scipy.optimize as sco
plt.rcParams['font.family'] = 'Arial Unicode MS'
plt.rcParams['axes.unicode_minus']=False

data = pd.read_csv('MPT.csv')
returns_annual = data.mean()*252#计算年化收益率
cov_annual = data.cov()*252#计算协方差矩阵

#模拟10000个投资组合
number_assets = 5
portfolio_returns = []#组合收益率
portfolio_volatilities = []#组合波动率
sharpe_ratio = []#夏普比率
for singel_portfolio in range(100000):
    weights = np.random.random(number_assets)#权重
    weights = weights/(np.sum(weights))#权重归一化
    returns = np.dot(weights,returns_annual)#投资组合收益率
    volatility = np.sqrt(np.dot(weights.T,np.dot(cov_annual,weights)))#投资组合波动率
    portfolio_returns.append(returns)
    portfolio_volatilities.append(volatility)
    sharpe = returns/volatility#计算夏普比率
    sharpe_ratio.append(sharpe)
portfolio_returns = np.array(portfolio_returns)
portfolio_volatilities = np.array(portfolio_volatilities)

plt.style.use('seaborn-dark')
plt.figure(figsize=(9, 5))
plt.scatter(portfolio_volatilities, portfolio_returns, c=sharpe_ratio,cmap='RdYlGn', edgecolors='black',marker='o')
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
#plt.savefig('/Users/harper/Desktop/2.png',dpi=500,bbox_inches = 'tight')
plt.show()

#绘制夏普比率
def statistics(weights):
    weights = np.array(weights)
    pret = np.sum(data.mean() * weights) * 252
    pvol = np.sqrt(np.dot(weights.T, np.dot(data.cov() * 252, weights)))
    return np.array([pret, pvol, pret / pvol])


def min_func_sharpe(weights):
    return -statistics(weights)[2]

#找出最优组合
bnds = tuple((0, 1) for x in range(number_assets))
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
opts = sco.minimize(min_func_sharpe, number_assets * [1. / number_assets, ], method='SLSQP', bounds=bnds,
                    constraints=cons)
print(opts['x'].round(6))  # 得到各股票权重
#print(statistics(opts['x']).round(3))  # 得到投资组合预期收益率、预期波动率以及夏普比率