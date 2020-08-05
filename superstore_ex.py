import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

df = pd.read_excel('superstore.xls')
print(df['Profit'].describe())

plt.scatter(range(df.shape[0]), np.sort(df['Profit'].values))
plt.xlabel('index')
plt.ylabel('Profit')
plt.title("Sales distribution")
sns.despine()

plt.show()

sns.distplot(df['Profit'])
plt.title("Distribution of Profit")
sns.despine()

plt.show()