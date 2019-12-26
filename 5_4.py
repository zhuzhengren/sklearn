import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

X_train = [[6],[8],[10],[14],[18]]
y_train = [[7],[9],[13],[17.5],[18]]
X_test = [[6],[8],[11],[16]]
y_test = [[8],[12],[15],[18]]

regressor = LinearRegression()
regressor.fit(X_train, y_train)

#linspace 0开始点 26结束点 100样本数量 
xx = np.linspace(0,26,100)
yy = regressor.predict(xx.reshape(xx.shape[0],1))
#显示线性预测的直线
plt.plot(xx, yy)

#对X值进行二次化处理 degree=2 表示2次
quadratic_featurizer = PolynomialFeatures(degree = 2)
X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
X_test_quadratic = quadratic_featurizer.transform(X_test)


#创建线性回归模型 进行训练
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(X_train_quadratic,y_train)

#通line_16 但是对数据进行二次化处理 
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))

#显示二项式曲线
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), c='r', linestyle = '--')

#显示标题 X,Y轴标签
plt.title('Pizza price  regressed on diameter')
plt.xlabel('Diameter in inches')
plt.ylabel('Price in dollars')

#设置X，y轴最大显示值
plt.axis([0,25,0,25])
#显示网格
plt.grid(True)
#显示散点
plt.scatter(X_train, y_train)

#显示整个画面
#plt.show()

print(X_train)
print(X_train_quadratic)
print(X_test)
print(X_test_quadratic)

print('simple line regression r-squared', regressor.score(X_test,y_test))

print('QUadratic regression r-squared', regressor_quadratic.score(X_test_quadratic,y_test))