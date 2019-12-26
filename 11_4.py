#简单使用SVM 对MNIST 进行分类

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_mldata

#
print('Data Reading...')
mnist = fetch_mldata('MNIST original', data_home='./datasets')
X, y = mnist.data, mnist.target

X = X/255.0*2-1
	
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11)

svm = SVC(kernel='rbf', gamma=0.03, C=100)

print('Data Readied')
print('Start Fit')

svm.fit(X_train[:10000],y_train[:10000])

print('Fit Completed')
print('Start Predict')	

predictions = svm.predict(X_test)

print(classification_report(y_test, predictions))

