#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.optimize import minimize

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures

def linearRegCostFunction(theta, X, y, lamb):
    m = y.size
    h = X.dot(theta)

    J = (1/(2*m))*np.sum(np.square(h-y)) + (lamb/(2*m))*np.sum(np.square(theta[1:]))

    return(J)

def linearGradientReg(theta, X, y, lamb):
    m = y.size
    h = X.dot(theta.reshape(-1, 1))

    grad = (1/m)*(X.T.dot(h-y))+ (lamb/m)*np.r_[[[0]],theta[1:].reshape(-1,1)]

    return(grad.flatten())

def trainLinearReg(X, y, lamb):
    initial_theta = np.array([[15],[15]])

    res = minimize(linearRegCostFunction, initial_theta, args=(X, y, lamb), method=None, jac=linearGradientReg, options={'maxiter':5000})

    return(res)

def learningCurve(X, y, Xval, yval, reg):
    m = y.size
    
    error_train = np.zeros((m, 1))
    error_val = np.zeros((m, 1))
    
    for i in np.arange(m):
        res = trainLinearReg(X[:i+1], y[:i+1], reg)
        error_train[i] = linearRegCostFunction(res.x, X[:i+1], y[:i+1], reg)
        error_val[i] = linearRegCostFunction(res.x, Xval, yval, reg)
    
    return(error_train, error_val)