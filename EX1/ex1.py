import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3dp import axes3d
from computeCost import computeCost
from gradientDescent import gradientDescent
from sklearn.linear_model import LinearRegression

pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_seq_items', None)


sns.set_context('notebook')
sns.set_style('white')

def warmUpExercise():
    return(np.identity(10))
    

warmUpExercise()

data = np.loadtxt('ex1data1.txt', delimiter=',')

X = np.c_[np.ones(data.shape[0]),data[:,0]]
y = np.c_[data[:,1]]

plt.scatter(X[:,1], y, s=50, c='r', marker='x', linewidths=1)
plt.xlim(4.5,26)

# name x and y
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')

#plt.show()

theta , Cost_J = gradientDescent(X, y)
print('theta: ',theta.ravel())

plt.plot(Cost_J)
plt.ylabel('Cost J')
plt.xlabel('Iterations')


#plt.show()

computeCost(X, y)
