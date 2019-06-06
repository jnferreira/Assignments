#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sigmoid

def lrcostFunctionReg(theta, lamb, X, y):

    m = y.size
    h = sigmoid(X.dot(theta))

    
