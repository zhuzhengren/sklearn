#使用决策树预测图片是不是广告

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV


df = pd.read_csv('./ad.data', header=None)


explanatory_variable_columns = set(df.columns.values)

#移除最后一列 结果列
explanatory_variable_columns.remove(len(df.columns.values)-1)

#取出最后一列 并且数字化 0，1
response_variable_cloumn = df[len(df.columns.values)-1]
y = [1 if e== 'ad.' else 0 for e in response_variable_cloumn]

#取出 X
X = df[list(explanatory_variable_columns)].copy()

#将缺省值使用-1 进行替换
X.replace(to_replace=' *?', value=-1, regex=True, inplace=True)

#拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y)

#网格搜索最优参数
plpeline = Pipeline([
	('clf', DecisionTreeClassifier(criterion='entropy'))
	])

parameters = {
	'clf__max_depth'		:	(150, 155, 160),
	'clf__min_samples_split':	(2, 3),
	'clf__min_samples_leaf'	:	(1, 2, 3)
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