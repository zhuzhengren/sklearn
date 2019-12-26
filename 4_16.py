import gensim

model = gensim.models.KeyedVectors.load_word2vec_format('')

embdding = model.word_vec('cat')

print("dimmensions: %s" % embdding.shape)

print(embdding)

print(model.most_similar(positive=['puppy','cat'], negative = ['kitten'],topn=1))

for i in model.most_similar(positive=['saddle','painter'],negative=['palette'],topn=3):
	print(i)