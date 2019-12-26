import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

df = pd.read_csv('/Users/zzr/Downloads/wine.csv')
X = df[list(df.columns)[1:-2]]
y = df['quality']

regressor = LinearRegression()
scores = cross_val_score(regressor, X, y, cv=5)
print(scores.mean())
print(scores)