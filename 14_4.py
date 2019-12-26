#使用PCA对人脸数据进行下降维度 然后使用逻辑回归区分数据

import os 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from PIL import Image

from sklearn.datasets import fetch_mldata
from sklearn.datasets import load_digits

X = []
y = []

'''
for dirpath, _, filenames in os.walk('att-faces/orl_faces'):
	for filename in filenames:
		if filename[-3:] == 'pgm':
			img = Image.open(os.path.join(dirpath,filename)).convert('L')
			arr = np.array(img).reshape(10304).astype('float32')/255.
			X.append(arr)
			y.append(dirpath)

'''

#data = load_digits()
data = fetch_mldata('MNIST original', data_home='./datasets')
X = data.data
y = data.target

X = scale(X)

X_train, X_test, y_train, y_test = train_test_split(X, y)
pca = PCA(n_components=64)

X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)
print(X_train.shape)
print(X_train_reduced.shape)

classifier = LogisticRegression()
accuracies = cross_val_score(classifier,X_train_reduced,y_train)

print(' Cross Validation Score: %s' % np.mean(accuracies))

classifier.fit(X_train_reduced, y_train)
predictions = classifier.predict(X_test_reduced)
print(classification_report(y_test, predictions))
