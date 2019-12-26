#使用 套袋法 创建随机森林

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report



'''
#创建人工分类数据集
n_samples 		1000个实例
n_features 100	 每个实例包含100个特征
n_informative 	100个特征中只有20个特征有意义 其他都是噪声 
'''
X, y = make_classification(n_samples=1000, n_features=100, n_informative=20, n_clusters_per_class=2, random_state=11)

#切分数据集 测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11)

#使用决策树进行判断
clf = DecisionTreeClassifier(random_state=11)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print(classification_report(y_test, predictions))


#使用包含10颗决策树的随机森林进行判断
clf = RandomForestClassifier(n_estimators=10, random_state=11)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(classification_report(y_test, predictions))
