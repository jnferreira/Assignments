#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sigmoid

from scipy.optimize import minimize


def lrcostFunctionReg(theta, lamb, X, y):
    m = y.size
    h = sigmoid.sigmoid(X.dot(theta))

    J = -1*(1/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y)) + \
        (lamb/(2*m))*np.sum(np.square(theta[1:]))

    return(J[0])


def lrgradientReg(theta, lamb, X, y):
    m = y.size
    h = sigmoid.sigmoid(X.dot(theta.reshape(-1, 1)))

    # voltar aqui
    grad = (1/m)*X.T.dot(h-y) + (lamb/m)*np.r_[[[0]], theta[1:].reshape(-1, 1)]

    return(grad.flatten())


def oneVsAll(features, classes, n_labels, reg):
    initial_theta = np.zeros((features.shape[1], 1))  # 401x1
    all_theta = np.zeros((n_labels, features.shape[1]))  # 10x401

    for c in np.arange(1, n_labels+1):
        res = minimize(lrcostFunctionReg, initial_theta, args=(reg, features, (classes == c)*1), method=None,
                       jac=lrgradientReg, options={'maxiter': 50})
        all_theta[c-1] = res.x

    return(all_theta)

def predictOneVsAll(all_theta, features):
    probs = sigmoid.sigmoid(features.dot(all_theta.T))

    return(np.argmax(probs, axis=1)+1)

