#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import functions

from scipy.optimize import minimize

from sklearn.preprocessing import PolynomialFeatures

pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_seq_items', None)
 
import seaborn as sns
sns.set_context('notebook')
sns.set_style('white')

data = functions.loaddata('ex2data1.txt', ',')

