#使用交叉验证 输出二元分类的的准确性,精准率 召回率 F1值

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

df= pd.read_csv('./SMSSpamCollection', delimiter = '\t', header=None)
X = df[1].values
y = df[0].values

for i,item in enumerate(y):
	y[i]=1 if item=='ham' else 0
y = y.astype('int')


X_train_raw, X_test_raw,y_train,y_test = train_test_split(X, y)
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)
classifier = LogisticRegression()
classifier.fit(X_train,y_train)

scores = cross_val_score(classifier, X_train, y_train, cv=5,scoring='accuracy')
print('Accuracies: %s' % scores)
print('Mean accuracy: %s' % np.mean(scores))


precisions = cross_val_score(classifier, X_train, y_train, cv=5,scoring='precision')
print('Precision:%s' % np.mean(precisions))
recalls = cross_val_score(classifier, X_train, y_train,cv = 5,scoring='recall')
print('Recall: %s' % np.mean(recalls))
f1s = cross_val_score(classifier, X_train, y_train,cv = 5,scoring='f1')
print('F1: %s' % np.mean(f1s))