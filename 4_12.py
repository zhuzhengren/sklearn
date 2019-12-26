from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag

wordnet_tags = ['n', 'v']
corpus = ['he ate the sandwiches',
		'every samdwich was eaten by him']

stemmer = PorterStemmer()
print('Stemmed:', [[stemmer.stem(token) for token in word_tokenize(document)] for document in corpus])

def lemmatize(token, tag):
	if tag[0].lower() in ['n','v']:
		return lemmatizer.lemmatize(token,tag[0].lower())
	return token

lemmatizer = WordNetLemmatizer()
tagged_corpus = [pos_tag(word_tokenize(document)) for document in corpus]
print('Lemmatized:', [[lemmatize(token,tag) for token, tag in document] for document in tagged_corpus])