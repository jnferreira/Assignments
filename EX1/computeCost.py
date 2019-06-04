import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def computeCost(X, y, theta=[[0],[0]]):
    m = y.size
    J = 0
    
    h = X.dot(theta)

    J = 1/(2*m)*np.sum(np.square(h-y))
    
    return(J)

