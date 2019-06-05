#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotdata
import sigmoid
import costFunGrad
import predict

from scipy.optimize import minimize

from sklearn.preprocessing import PolynomialFeatures

pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_seq_items', None)

import seaborn as sns
sns.set_context('notebook')
sns.set_style('white')

data = np.loadtxt('ex2data1.txt', delimiter=',')

X = np.c_[np.ones((data.shape[0],1)), data[:,0:2]]
y = np.c_[data[:,2]]

#print (np.ones((data.shape[0],1)), data[:,0:2])

#plotdata.plotData(data, 'Exam 1 score', 'Exam 2 score', 'Admitted', 'Not admitted')

initial_theta = np.zeros(X.shape[1])

cost = costFunGrad.costFunction(initial_theta, X, y)
grad = costFunGrad.gradient(initial_theta, X, y)

print('Cost ', cost)
print('Gradient ', grad)

res = minimize(costFunGrad.costFunction, initial_theta, args=(X,y), method=None, jac=costFunGrad.gradient, options={'maxiter':400})
#Gives the best theta
#print(res.x)

#print(sigmoid.sigmoid(np.array([1, 85, 85]).dot(res.x.T)))

p = predict.predict(res.x, X)

print('Train accuracy {}%'.format(100*sum(p == y.ravel())/p.size))

x1_min, x1_max = X[:,1].min(), X[:,1].max(),
x2_min, x2_max = X[:,2].min(), X[:,2].max(),
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

plt.scatter(45, 85, s=60, c='r', marker='v', label='(45, 85)')
h = sigmoid.sigmoid(np.c_[np.ones((xx1.ravel().shape[0],1)), xx1.ravel(), xx2.ravel()].dot(res.x))
h = h.reshape(xx1.shape)
plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
plotdata.plotData(data, 'Exam 1 score', 'Exam 2 score', 'Admitted', 'Not admitted')
