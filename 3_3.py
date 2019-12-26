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


lb = LabelBinarizer()
y_train_binarized = lb.fit_transform(y_train)


K = 3
clf = KNeighborsClassifier(n_neighbors=K)
clf.fit(X_train, y_train_binarized.reshape(-1))

prediction_binarized = clf.predict(np.array([155,70]).reshape(1,-1))[0]

predict_label = lb.inverse_transform(prediction_binarized)
