from sklearn import datasets

digits = datasets.load_digits()
print("Digits: %s " % digits.target[0])
print(digits.images[0])
print('Feature vector:\n %s ' % digits.images[0].reshape(-1,64))