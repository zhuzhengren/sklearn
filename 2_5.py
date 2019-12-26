import  numpy as np

X = np.array([[6],[8],[10],[14],[18]]).reshape(-1,1)

print(np.var(X, ddof=1))