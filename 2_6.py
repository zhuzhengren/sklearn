import  numpy as np

X = np.array([[6],[8],[10],[14],[18]]).reshape(-1,1)
y = np.array([7,9,13,17.5,18])

x_bar = X.mean()
y_bar = y.mean()

covariance = np.multiply((X - x_bar).transpose(), y - y_bar).sum()/(X.shape[0]-1)

print(covariance)

print(np.cov(X.transpose(), y )[0][1])

