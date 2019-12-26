#停用词过滤 删除部分无意义的单词 此处为in the
corpus = [
	'NUC played Duke in basketball',
	'Duke lost the basketball game'
]


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)