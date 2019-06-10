#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import functions

from scipy.io import loadmat
from sklearn.svm import SVC

pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_seq_items', None)


import seaborn as sns
sns.set_context('notebook')
sns.set_style('white')

data1 = loadmat('ex6data1.mat')
data1.keys()

y1 = data1['y']
X1 = data1['X']

#print('X1:', X1.shape)
#print('y1:', y1.shape)

clf = SVC(C=1.0, kernel='linear')
clf.fit(X1, y1.ravel())
#functions.plot_svc(clf, X1, y1)

clf.set_params(C=100)
clf.fit(X1, y1.ravel())
#functions.plot_svc(clf, X1, y1)

x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2

#print(functions.gaussianKernel(x1, x2, sigma))

data2 = loadmat('ex6data2.mat')
data2.keys()

y2 = data2['y']
X2 = data2['X']

print('X2:', X2.shape)
print('y2:', y2.shape)

clf2 = SVC(C=50, kernel='rbf', gamma=6)
clf2.fit(X2, y2.ravel())
#functions.plot_svc(clf2, X2, y2)

data3 = loadmat('ex6data3.mat')
data3.keys()

y3 = data3['y']
X3 = data3['X']

print('X3:', X3.shape)
print('y3:', y3.shape)

clf3 = SVC(C=1.0, kernel='poly', degree=3, gamma=10)
clf3.fit(X3, y3.ravel())
functions.plot_svc(clf3, X3, y3)

data4 = pd.read_table('vocab.txt', header=None)
data4.info()

print(data4.head())

