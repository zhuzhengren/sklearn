import  numpy as np


X = np.array([[6],[8],[10],[14],[18]]).reshape(-1,1)

y = [7,9,13,17.5,18]


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X ,y)

test_pizza = np.array([[12]])
predicted_price = model.predict(test_pizza)[0]

print('Residual sum of squares: %.2f' % np.mean((model.predict(X)-y)  **2))