#使用网格搜索找到多元分类中最优参数解


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

import pandas as pd
import numpy as np

df = pd.read_csv('./train.tsv', header=0, delimiter='\t')
X ,y = df['Phrase'], df['Sentiment'].as_matrix()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)

#grid_search = main(X_train,  y_train)

pipeline = Pipeline([
	('vect',TfidfVectorizer(stop_words='english')),
	('clf',LogisticRegression())])

parameters = {
	'vect__max_df'		:	(0.25, 0.5),
	'vect__ngram_range'	:	((1,1), (1,2)),
	'vect__use_idf'		:	(True, False),
	'clf__C'			:	(0.1, 1, 10)
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs = -1,verbose=1, scoring='accuracy')

grid_search.fit (X_train, y_train)
print('Best score: %3f' % grid_search.best_score_)
print('Best parameters set: ')

best_parameters = grid_search.best_estimator_.get_params()

for param_name in sorted(parameters.keys()):
	print('\t %s: %r' % (param_name, best_parameters[param_name]))
