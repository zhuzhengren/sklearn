from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier

import numpy as np

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

X_test = np.array([
	[168,65],
	[180,96],
	[160,52],
	[169,67]
	])
y_test = ['male','male','female','female']



lb = LabelBinarizer()
y_train_binarized = lb.fit_transform(y_train)
K = 3
clf = KNeighborsClassifier(n_neighbors=K)
clf.fit(X_train, y_train_binarized.reshape(-1))


y_test_binarized = lb.transform(y_test)

print('Binarized labels: %s' % y_test_binarized.T[0])

predictions_binarized  = clf.predict(X_test)
print('Predicted labels: %s' % lb.inverse_transform(predictions_binarized))

from sklearn.metrics import accuracy_score
print('Accuracy: %s' % accuracy_score(y_test_binarized,predictions_binarized))