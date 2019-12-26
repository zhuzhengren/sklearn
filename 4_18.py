import os 
import caffe
import numpy as np
CAFFE_DIR = "/usr/local/Cellar/caffe/1.0_16/"

MEAN_PATH = os.path.join(CAFFE_DIR,'python/caffe/imagenet/ilsvrc_2012_mean.npy')
PROTOTXT_PATH = os.path.join(CAFFE_DIR,'models/bvlc_reference_caffenet/deploy.prototxt')
CAFFEMODEL_PATH = os.path.join(CAFFE_DIR,'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')

IMAGE_PATH = '/Users/zzr/Desktop/obama.jpg'

net = caffe.Net(PROTOTXT_PATH, CAFFEMODEL_PATH,caffe.TEST)

transformer = caffe.io.transformer({'data':net.blobs['data'].data.shape})
transformer.set_transpose('data',(2,0,1))
transformer.set_mean('data',np.load(MEAN_PATH).mean(1).mean(1))
transformer.set_raw_scale('data',255)
transformer.set_channel_swap('data',(2,1,0))

net.blobs['data'].reshape(1,3,227,227)
net.blobs['data'].data[0] = transformer.preprocess('data',caffe.io.load_image(IMAGE_PATH))

net.forward()

features = net.blobs['fc7'].data.reshape(-1,)
print(features.shape)
print(features)