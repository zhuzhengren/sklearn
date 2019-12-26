#词袋模型 
corpus = [
	'NUC played Duke in basketball',
	'Duke lost the basketball game'
]


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)