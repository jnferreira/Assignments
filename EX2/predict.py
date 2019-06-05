#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sigmoid

def predict(theta, X):
    p = sigmoid.sigmoid(X.dot(theta.T)) >= 0.5
    return(p)