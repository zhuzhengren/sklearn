#不实用词干 同义但是距离较远
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import euclidean_distances
corpus = [
	'He ate the sandwiches ',
	'every sandwich was eaten by him'
]

vectorizer = CountVectorizer(binary = True,stop_words='english')
X=vectorizer.fit_transform(corpus).todense()
print(vectorizer.vocabulary_)
print(X)
print('distance between 1st and 2nd documents:' ,euclidean_distances(X[0],X[1]))
