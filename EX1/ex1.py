#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import axes3d

import computeCost
import gradientDescent
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

# plt.scatter(X[:,1], y, s=50, c='r', marker='x', linewidths=1)
# plt.xlim(4.5,26)

# # name x and y
# plt.xlabel('Population of City in 10,000s')
# plt.ylabel('Profit in $10,000s')

theta , Cost_J = gradientDescent.gradientDescent(X, y)
# print('theta: ',theta.ravel())

# plt.plot(Cost_J)
# plt.ylabel('Cost J')
# plt.xlabel('Iterations')
# plt.show()


xx = np.arange(5,23)
yy = theta[0]+theta[1]*xx

# Plot gradient descent
plt.scatter(X[:,1], y, s=30, c='r', marker='x', linewidths=1)
plt.plot(xx,yy, label='Linear regression (Gradient descent)')

# Compare with Scikit-learn Linear regression 
regr = LinearRegression()
regr.fit(X[:,1].reshape(-1,1), y.ravel())
plt.plot(xx, regr.intercept_+regr.coef_*xx, label='Linear regression (Scikit-learn GLM)')

plt.xlim(4,24)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.legend(loc=4)
plt.show()


