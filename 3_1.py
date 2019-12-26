import numpy as np
import matplotlib.pyplot as plt

X_train = np.array([
	[158,64],
	[170,86],
	[184,84],
	[191,80],
	[155,49],
	[163,59],
	[180,67],
	[158,54],
	[170,67]
	])

y_train = ['male','male','male','male','female','female','female','female','female']

plt.figure()
plt.title("Human Height and Weight by Sex")
plt.xlabel('Height in cm')
plt.ylabel('Weight in Kg')

for i,x in enumerate(X_train):
	plt.scatter(x[0],x[1],c='k',marker='x' if y_train[i]=='male' else 'D')

plt.grid(True)
plt.show()