#特征值的维度对模型结果的影响
import numpy as np
from scipy.spatial.distance import euclidean

X_train = np.array([
	[1700,1],[1600,0]
	])
x_test = np.array([1640, 1]).reshape(1,-1)
print(euclidean(X_train[0,:], x_test))
print(euclidean(X_train[1,:],x_test))

X_train = np.array([
	[1.7,1],[1.6,0]
	])
x_test = np.array([164, 1]).reshape(1,-1)
print(euclidean(X_train[0,:], x_test))
print(euclidean(X_train[1,:],x_test))
