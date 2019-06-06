#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sigmoid
import functions

# load MATLAB files
from scipy.io import loadmat
from scipy.optimize import minimize

from sklearn.linear_model import LogisticRegression

pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_seq_items', None)

import seaborn as sns
sns.set_context('notebook')
sns.set_style('white')

data = loadmat('ex3data1.mat')
weights = loadmat('ex3weights.mat')

#print(weights['Theta1'])
#print(data['y'])

y = data['y']
# add a column with ones
X = np.c_[np.ones((data['X'].shape[0],1)), data['X']]

#print('X: {} (with intercept)'.format(X.shape))
#print('y: {}'.format(y.shape))

theta1, theta2 = weights['Theta1'], weights['Theta2']

#print('Theta1: ', theta1.shape)
#print('Theta2: ', theta2.shape)

sample = np.random.choice(X.shape[0], 20)
plt.imshow(X[sample,1:].reshape(-1,20).T)
plt.axis('off')
#plt.show()


theta = functions.oneVsAll(X, y, 10, 0.1)

pred = functions.predictOneVsAll(theta, X)

print('Training set accurancy: {} %'.format(np.mean(pred == y.ravel())*100))

# Multiclass Logistic Regression with scikit-learn


#clf = LogisticRegression(C=10, penalty='12', solver='liblinear')

#clf.fit(X[:,1:],y.ravel())

#pred2 = clf.predict(X[:,1:])
#print('Training set accuracy: {} %'.format(np.mean(pred2 == y.ravel())*100))

def predict(theta_1, theta_2, features):
    z2 = theta_1.dot(features.T)
    a2 = np.c_[np.ones((data['X'].shape[0],1)), sigmoid.sigmoid(z2).T]
    
    z3 = a2.dot(theta_2.T)
    a3 = sigmoid.sigmoid(z3)
        
    return(np.argmax(a3, axis=1)+1) 


pred = predict(theta1, theta2, X)
print('Training set accuracy: {} %'.format(np.mean(pred == y.ravel())*100))

