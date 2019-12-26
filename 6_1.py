#测试数据集
import pandas as pd

df= pd.read_csv('./SMSSpamCollection', delimiter = '\t', header=None)
print(df.head())

print('Number of spam messages:%s' % df[df[0]=='spam'][0].count())

print('Number of ham messages:%s' % df[df[0]=='ham'][0].count())