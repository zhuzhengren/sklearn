#图像量化 使用一种颜色替换一系列颜色
#使用K-means方法 对颜色进行分类 达到压缩图片的目的

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from PIL import Image

#读取图片 除以255 归一化 例如原颜色 (150,230,50) -> (0.588,0.902,0.196)
original_img = np.array(Image.open('datasets/dog.jpeg'), dtype=np.float64)/255
#original_dimensions = tuple(original_img.shape)

#获取图片的 宽度 高度 深度
width, height, depth = tuple(original_img.shape)

#将数组按照 宽高深 格式化
image_flattened = np.reshape(original_img,(width*height, depth))

print(image_flattened[1])

#随机选取1000种颜色 将1000种颜色分类为64种
image_array_sample = shuffle(image_flattened, random_state=0)[:1000]

estimator = KMeans(n_clusters=64, random_state=0)
estimator.fit(image_array_sample)

#同上等效代码
#KMeans(algorithm='auto', copy_x = True, init='k-means++', max_iter=300, n_clusters=64, 
#	n_init=10, n_jobs=1, precompute_distances='auto', random_state=0, tol=0.0001, verbose=0)


#将原图片按照分类规则重新划定颜色
cluster_assignments = estimator.predict(image_flattened)

#将结果取出赋值到compressed_img中
compressed_palette = estimator.cluster_centers_
compressed_img = np.zeros((width, height, compressed_palette.shape[1]))
label_idx = 0
for i in range(width):
	for j in range(height):
		compressed_img[i,j] = compressed_palette[cluster_assignments[label_idx]]
		label_idx += 1
#绘图 原图
plt.subplot(1,2,1)
plt.title('Original Image', fontsize=24)
plt.imshow(original_img)
plt.axis('off')
#压缩图
plt.subplot(1,2,2)
plt.title('Compressed Image', fontsize=24)
plt.imshow(compressed_img)
plt.axis('off')
plt.show()