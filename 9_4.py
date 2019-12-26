#使用堆叠法创建决策集合 每个决策元素可以不仅仅是决策树 也可以是KNN 或者逻辑回归等任何方式

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.base import clone, BaseEstimator, TransformerMixin, ClassifierMixin

#定义一个基准决策
class StackingClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
	def __init__(self, classifiers):
		self.classifiers = classifiers
		self.meta_classifier = DecisionTreeClassifier()

	def fit(self, X, y):
		for clf in self.classifiers:
			clf.fit(X, y)
		self.meta_classifier.fit(self._get_meta_features(X), y)
		return self

	def _get_meta_features(self, X):
		probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers])
		return np.concatenate(probas, axis=1)

	def predict(self, X):
		return self.meta_classifier.predict(self._get_meta_features(X))

	def predict_proba(self, X):
		return self.meta_classifier.predict_proba(self._get_meta_features(X))

#创建数据集
X, y = make_classification(n_samples=1000, n_features=50, n_informative=30, n_clusters_per_class=3, random_state=11)
#拆分测试集和训练集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11)

#使用逻辑回归预测
lr = LogisticRegression()
lr.fit(X_train, y_train)
print('Logistic regression accuracy: %s' % lr.score(X_test, y_test))

#使用KNN预测
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
print('KNN accuracy: %s' % knn_clf.score(X_test, y_test))

#使用决策树进行预测
dtc = DecisionTreeClassifier(random_state=11)
dtc.fit(X_train, y_train)
print('Decision Tree accuracy: %s' % dtc.score(X_test, y_test))

#使用**预测
base_classifiers = [lr, knn_clf]
stacking_clf = StackingClassifier(base_classifiers)
stacking_clf.fit(X_train, y_train)
print('Stacking classifier accuracy : %s' % stacking_clf.score(X_test, y_test))