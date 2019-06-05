#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sigmoid

def costFunction(theta, X, y):
    m = y.size
    h = sigmoid.sigmoid(X.dot(theta))

    J = -(1 / m) * ((np.log(h).T.dot(y)) + np.log(1 - h).T.dot(1-y))

    return (J[0])

def gradient(theta, X, y):
    m = y.size
    h = sigmoid.sigmoid(X.dot(theta.reshape(-1,1)))

    g = (1 / m) * X.T.dot(h - y)

    return(g.flatten())


# *args - don't know how many arguments are passed
def costFunctionReg(theta, lamb, XX, y):

    m = y.size
    h = sigmoid.sigmoid(XX.dot(theta))

    J = -(1 / m) * ((np.log(h).T.dot(y)) + np.log(1 - h).T.dot(1-y)) + (lamb/(2*m))*np.sum(np.square(theta[1:]))

    return(J[0])

def gradientReg(theta, lamb, XX, y):
    m = y.size
    h = sigmoid.sigmoid(XX.dot(theta.reshape(-1,1)))

    g = (1 / m) * XX.T.dot(h - y) + (lamb/(m))*np.sum(np.square(theta[1:]))

    return(g.flatten())
    

