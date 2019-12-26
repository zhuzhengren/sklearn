#使用网格搜索查找随机森林最优的数

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report


from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV


'''
#创建人工分类数据集
n_samples 		1000个实例
n_features 100	 每个实例包含100个特征
n_informative 	100个特征中只有20个特征有意义 其他都是噪声 
'''
X, y = make_classification(n_samples=1000, n_features=100, n_informative=20, n_clusters_per_class=2, random_state=11)

#切分数据集 测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11)





#网格搜索最优参数
plpeline = Pipeline([
	('clf', RandomForestClassifier())
	])

parameters = {
	'clf__n_estimators'	:	(10, 20, 30, 50, 100)
} 

grid_search = GridSearchCV(plpeline, parameters, n_jobs=-1, verbose=1, scoring='f1')

#训练
grid_search.fit(X_train, y_train)





#取出最优参数集
best_parameters = grid_search.best_estimator_.get_params()
print('Best score: %0.3f' % grid_search.best_score_)
print('Best parameters set:')
for param_name in sorted(parameters.keys()):
	print('\t%s : %r' % (param_name, best_parameters[param_name]))

#预测测试集 并进行打分
predictions = grid_search.predict(X_test)
print(classification_report(y_test, predictions))

