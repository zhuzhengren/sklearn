#使用hash方法直接遍历一遍产生特征向量
from sklearn.feature_extraction.text import HashingVectorizer

corpus = ['the', 'ate', 'bacon', 'cat']

#n_features默认2^20 特征向量的长度
vectorizer = HashingVectorizer(n_features=6)

print(vectorizer.transform(corpus).todense())