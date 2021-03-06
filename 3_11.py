#KNN 回归 使用身高和性别来判断体重
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X_train = np.array([
	[158,1],
	[170,1],
	[183,1],
	[191,1],
	[155,0],
	[163,0],
	[180,0],
	[158,0],
	[170,0]
	])
y_train = [64,86,84,80,49,59,67,54,67]

X_test = np.array([
	[168,1],
	[180,1],
	[160,0],
	[169,0]
	])
y_test = [65,96,52,67]

K = 3
clf = KNeighborsRegressor(n_neighbors = K)
clf.fit(X_train,y_train)

preditions = clf.predict(X_test)
print('Predicted weights: %s' % preditions)

print('Coefficient of determination: %s ' % r2_score(y_test,preditions))

print('Mean absolute error: %s' % mean_absolute_error(y_test,preditions))

print('Mean squared error: %s' % mean_squared_error(y_test,preditions))