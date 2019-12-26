#numpy 计算协方差

import numpy as np

X = np.array([
	[2,   0, -1.4],
	[2.2, 0.2, -1.5],
	[2.4, 0.1, -1],
	[1.9, 0, -1.2]
	])
print(np.cov(X).T)