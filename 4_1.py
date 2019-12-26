#对类别特征进行one-hot转换
from sklearn.feature_extraction import DictVectorizer
onehot_encoder = DictVectorizer()

X = [
	{'city':'New York'},
	{'city':'San Francisco'},
	{'city':'CHapel Hill'}
]

print(onehot_encoder.fit_transform(X).toarray())