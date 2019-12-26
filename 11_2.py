#使用SVM 方式对MNIST进行分类 并使用网格搜索找到SVM最优参数

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original', data_home='./datasets')


if __name__  == '__main__':
	X, y = mnist.data, mnist.target
	X = X/255.0*2 -1
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11)

	pipeline = Pipeline([
		('clf', SVC(kernel='rbf', gamma=0.01, C=100))
		])
	parameters = {
	 'clf__gamma'	:	(0.01, 0.03, 0.1, 0.3, 1),
	 'clf__C'		:	(0.1, 0.3, 1, 3, 10, 30)
	}

	grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1,verbose=1, scoring='accuracy')
	grid_search.fit(X_train[:10000], y_train[:10000])
	print('Best parameters set:')
	bset_parameters = grid_search.best_estimator_.get_params()
	for param in sorted(parameters.keys()):
		print('\t %s : %r' % (param, bset_parameters[param]))

	predictions = grid_search.predict(X_test)
	print(classification_report(y_test, predictions))

