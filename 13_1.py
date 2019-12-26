#实用 肘部法 确定 K-Means 中的K

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

c1x = np.random.uniform(0.5, 1.5,(1, 10))
c1y = np.random.uniform(0.5, 1.5,(1, 10))
c2x = np.random.uniform(3.5, 4.5,(1, 10))
c2y = np.random.uniform(3.5, 3.5,(1, 10))
x = np.hstack((c1x, c2x))
y = np.hstack((c1y, c2y))
X = np.vstack((x,y)).T

K = range(1,10)

meanDispersions = []
for k in K:
	kmeans = KMeans(n_clusters=k)
	kmeans.fit(X)
	meanDispersions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1))/X.shape[0])


plt.plot(K, meanDispersions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average Dispersion')
plt.title('Selection k with the Elbow Method')
plt.show()
