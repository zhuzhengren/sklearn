#使用感知机 对20newsgroups 数据集进行分类

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import f1_score, classification_report

categories = ['rec.sport.hockey', 'rec.sport.baseball', 'rec.autos']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, remove=('header', 'footer', 'quotes'))
newsgroups_test  = fetch_20newsgroups(subset='test',  categories=categories, remove=('header', 'footer', 'quotes'))

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(newsgroups_train.data)
X_test = vectorizer.transform(newsgroups_test.data)

clf = Perceptron(random_state=11)
clf.fit(X_train, newsgroups_train.target)

predictions = clf.predict(X_test)
print(classification_report(newsgroups_test.target, predictions))