#14_3 高伟数据可视化。手工计算特征值 特征向量

#使用PCA对高维数据可视化

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import numpy as np

data = load_iris()
y = data.target
X = data.data


#减去平均数
X = X - X.mean()

#计算协方差
cov_X = np.cov(X.T)

#计算协方差的特征值 和特征向量
w, v = np.linalg.eig(cov_X)


#提取前两列的特征向量
v_2 = (v.T[:2]).T

#将数据 和特征向量相乘 压缩
reduced_X = np.dot(X,v_2)


#绘图
red_x, red_y =[], []
blue_x, blue_y = [], []
green_x, green_y = [],[]

for i in range(len(reduced_X)):
	if y[i]==0:
		red_x.append(reduced_X[i][0])
		red_y.append(reduced_X[i][1])
	elif y[i]==1:
		blue_x.append(reduced_X[i][0])
		blue_y.append(reduced_X[i][1])
	else :
		green_x.append(reduced_X[i][0])
		green_y.append(reduced_X[i][1])

plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()