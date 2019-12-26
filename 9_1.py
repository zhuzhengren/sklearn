import numpy as np

sample = np.random.randint(low=1, high=100,size=10)
print('Original sample: %s ' % sample)
print('Sample mean: %s' % sample.mean())

resamples = [np.random.choice(sample, size=sample.shape) for i in range(100)]
print('Number of bootstrap re-samples: %s' % len(resamples))
print('Example re-samples: %s' % resamples[0])

resample_means = np.array([resample.mean() for resample in resamples])
print('Mean of re-samples\' means: %s' % resample_means.mean())