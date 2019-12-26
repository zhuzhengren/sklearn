#使用AdaBoot创建随机森林


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X,  y = make_classification(n_samples=1000, n_features=50, n_informative=30, n_clusters_per_class=3, random_state=11)

X_train, X_test, y_train, y_test = train_test_split(X, y)

clf = DecisionTreeClassifier(random_state=11)
clf.fit(X_train, y_train)
print('Decision tree accuracy: %s ' % clf.score(X_test, y_test))

clf = AdaBoostClassifier(n_estimators=500, random_state=11)
clf.fit(X_train, y_train)

accuracies= []
accuracies.append(clf.score(X_test, y_test))

plt.title('Ensemble Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of base estimators in ensemble')

#**********?????????????????*********************** 不懂
plt.plot(range(1,501),[accuracy for accuracy in clf.staged_score(X_test, y_test)])

plt.show()
