import pandas as pd
df = pd.read_csv('/Users/zzr/Downloads/wine.csv')

import matplotlib.pyplot as plt

plt.scatter(df['alcohol'], df['quality'])

plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.title('Alcohol Against Quality')
plt.show()