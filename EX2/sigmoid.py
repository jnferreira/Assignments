#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def sigmoid(z):
    return(1 / (1 + np.exp(-z)))