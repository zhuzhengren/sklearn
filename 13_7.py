#通过聚类学习特征
#
import numpy as np
import os
#import mahotas as mh
#from mahotas.features import surf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.cluster import MiniBatchKMeans

import glob

all_instance_filenames = []
all_instance_targets = []

for f in glob.glob('datasets/dogs-and-cats/*/*.jpg'):
	target = 1 if 'Cat' in os.path.split(f)[0] else 0
	all_instance_filenames.append(f)
	all_instance_targets.append(target)

surf_features = []
for f in all_instance_filenames:
	image = mh.imread(f, as_grey = True)
	surf_features.append(surf.surf(image)[:, 5:])

#切分训练集 测试集
train_len = int(len(all_instance_filenames) *0.60)
X_train_surf_features = np.concatenate(surf_features[:train_len])
X_test_surf_features  = np.concatenate(surf_features[train_len:])
y_train = all_instance_targets[:train_len]
y_test = all_instance_targets[train_len:]

#开始训练
n_clusters = 300
estimator = MiniBatchKMeans(n_clusters = n_clusters)
estimator.fit_transform(X_train_surf_features)

#构建训练集
X_train = []
for instance in surf_features[:train_len]:
	clusters = estimator.predict(instance)
	features = np.bitcount(clusters)
	if len(features) < n_clusters:
		features = np.append(features, np.zeros((1, n_clusters - len(features))))
	X_train.append(features)

#构建测试集
X_test = []
for instance in surf_features[train_len:]:
	clusters = estimator.predict(instance)
	features = np.bitcount(clusters)
	if len(features) < n_clusters:
		features = np.append(features, np.zeros((1, n_clusters - len(features))))
	X_test.append(features)

#使用逻辑回归进行训练 并计算分数
clf = LogisticRegression(C=0.001, penalty='l2')
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(classification_report(y_test, predictions))

