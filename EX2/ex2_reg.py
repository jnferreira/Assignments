#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotdata
import costFunGrad

from scipy.optimize import minimize

from sklearn.preprocessing import PolynomialFeatures

pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_seq_items', None)

import seaborn as sns
sns.set_context('notebook')
sns.set_style('white')

data = np.loadtxt('ex2data2.txt', delimiter=',')

X = data[:,0:2]
y = np.c_[data[:,2]]


#plotdata.plotData(data, 'Microchip Test1', 'Microchip Test2', 'y = 1', 'y = 0')

# Note that this function inserts a column with 'ones' in the design matrix for the intercept.
poly = PolynomialFeatures(6)
XX = poly.fit_transform(data[:,0:2])
XX.shape
initial_theta = np.zeros(XX.shape[1])

print(costFunGrad.gradientReg(initial_theta, 1, XX, y))



