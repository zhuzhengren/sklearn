#使用交叉验证 输出二元分类的的准确性
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
X_train_raw, X_test_raw,y_train,y_test = train_test_split(X, y)
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)
classifier = LogisticRegression()
classifier.fit(X_train,y_train)

scores = cross_val_score(classifier, X_train, y_train, cv=5)
print('Accuracies: %s' % scores)
print('Mean accuracy: %s' % np.mean(scores))