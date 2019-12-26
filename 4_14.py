#使用tf_idf方法表示单词权重

from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ['The dog ate a sandwich and I ate a sandwich',
		'the wizard transfigured a sandwich']

vectorizer = TfidfVectorizer(stop_words = 'english')

tfidf_frequences = vectorizer.fit_transform(corpus).todense()

print(tfidf_frequences)

print('Token indices %s' % vectorizer.vocabulary_)