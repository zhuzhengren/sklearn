import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

corpus = ['The dog ate a sandwich,the wizard transfigured a sandwich, and I ate a sandwich']
vectorizer = CountVectorizer(stop_words = 'english')
frequencies = np.array(vectorizer.fit_transform(corpus).todense())[0]
print(frequencies)

print('Token indices %s' % vectorizer.vocabulary_)
for token, index in vectorizer.vocabulary_.items():
	print('The token "%s" appears %s times' % (token,frequencies[index]))