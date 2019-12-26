from sklearn.linear_model import  LinearRegression
import pandas as pd
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('/Users/zzr/Downloads/wine.csv')
X = df[list(df.columns)[1:-2]]
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y)

regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_predictions = regressor.predict(X_test)
print('R-squared: %s ' % regressor.score(X_test, y_test))