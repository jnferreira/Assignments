import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
#from mpl_toolkits.mplot3dp import axes3d
from computeCost import computeCost

pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_seq_items', None)


import seaborn as sns
sns.set_context('notebook')
sns.set_style('white')

def warmUpExercise():
    return(np.identity(10))
    

warmUpExercise()

data = np.loadtxt('ex1data1.txt', delimiter=',')

X = np.c_[np.ones(data.shape[0]),data[:,0]]
y = np.c_[data[:,1]]

plt.scatter(X[:,1], y, s=30, c='r', marker='x', linewidths=1)
plt.xlim(4,24)

# name x and y
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')

# plt.show()

print(computeCost(X, y))



