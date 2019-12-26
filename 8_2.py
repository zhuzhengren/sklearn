#使用决策树预测 分类垃圾邮件

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv('./SMSSpamCollection', delimiter='\t', header=None)
X = df[1].values 
y = df[0].values


#拆分训练集和测试集
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y)


vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)


dtc = DecisionTreeClassifier(criterion='entropy')


#训练
dtc.fit(X_train, y_train)


predictions = dtc.predict(X_test)
print(classification_report(y_test, predictions))

X_test = ['''Even my brother is not like to speak with me Is that seriously how you spell his name?''']
print(dtc.predict(vectorizer.transform(X_test)))

X_test = ['''07732584351 - Rodger Burns - MSG = We tried to call you re your reply to our sms for a free nokia mobile + free camcorder 
Correct or Incorrect? End? Reply END SPTV''']
print(dtc.predict(vectorizer.transform(X_test)))