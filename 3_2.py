import numpy as np
import matplotlib.pyplot as plt

X_train = np.array([
	[158,64],
	[170,86],
	[184,84],
	[191,80],
	[155,49],
	[163,59],
	[180,67],
	[158,54],
	[170,67]
	])

y_train = ['male','male','male','male','female','female','female','female','female']


x = np.array([[155,70]])

distances = np.sqrt(np.sum((X_train - x)**2, axis = 1))

nearest_neighbor_indices = distances.argsort()[:3]
nearest_neighbor_genders = np.take(y_train, nearest_neighbor_indices)


from collections import Counter
b = Counter(np.take(y_train,distances.argsort()[:3]))


print(b.most_common(1)[0][0])
