#numpy 计算特征值 和特征向量
import numpy as np

w, v = np.linalg.eig(np.array([[1,-2],[2,-3]]))

print(w)
print(v)

X = np.array([
	[2,   0, -1.4],
	[2.2, 0.2, -1.5],
	[2.4, 0.1, -1]
	])

a, b = np.linalg.eig(X)

print(a)
print(b)