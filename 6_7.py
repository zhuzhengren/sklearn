#使绘图显示召回率

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


#二次分类只能定义为binary 不能式字符串 将y进行简单的处理
for i,item in enumerate(y):
	y[i]=1 if item=='ham' else 0
#？？直接输入0或1 数据类型为object		
#print(y.dtype)
#将数据类型转换为int
y = y.astype('int')


#数据预处理 数据集测试集分割
X_train_raw, X_test_raw,y_train,y_test = train_test_split(X, y)
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

#定义分类器 并进行训练
classifier = LogisticRegression()
classifier.fit(X_train,y_train)

#对测试集进行预测
predictions  = classifier.predict_proba(X_test)

#计算 假正率 召回率 阈值
false_positive_rate, recall, thresholds = roc_curve(y_test,predictions[:, 1])

#计算ROC
roc_auc = auc(false_positive_rate, recall)

#绘图
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, recall,'b', label='AUC = %0.2f' %roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.show()
