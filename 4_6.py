#欧几里得范数
corpus = [
	'NUC played Duke in basketball',
	'Duke lost the basketball game',
	'I ate a sandwich'
]


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

from sklearn.metrics.pairwise import euclidean_distances
X = vectorizer.fit_transform(corpus).todense()
print('distance between 1st and 2nd documents:' ,euclidean_distances(X[0],X[1]))
print('distance between 2nd and 3rd documents:' ,euclidean_distances(X[1],X[2]))
print('distance between 3rd and 1st documents:' ,euclidean_distances(X[0],X[2]))