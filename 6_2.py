#使用逻辑回归实现二元分类
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

df= pd.read_csv('./SMSSpamCollection', delimiter = '\t', header=None)
X = df[1].values
y = df[0].values
X_train_raw, X_test_raw,y_train,y_test = train_test_split(X, y)
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)
classifier = LogisticRegression()
classifier.fit(X_train,y_train)
predictions = classifier.predict(X_test)
for i ,prediction in enumerate(predictions[:5]):
	print('Predicted:%s, message: %s' % (prediction, X_test_raw[i]))
