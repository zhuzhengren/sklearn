import  numpy as np
import matplotlib.pyplot as plt


X = np.array([[6],[8],[10],[14],[18]]).reshape(-1,1)

y = [7,9,13,17.5,18]

plt.figure()
plt.title('Pizza price plotted giantst diameter')
plt.xlabel('Diameter in inches')
plt.ylabel('Price in dollars')
plt.plot(X,y,'k.')
plt.axis([0,25,0,25])
plt.grid(True)
plt.show()


