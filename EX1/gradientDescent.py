import numpy as np
from computeCost import computeCost

def gradientDescent(X, y, theta=[[0],[0]], alpha=0.01, num_iters=1500):

    m = y.size
    J_history = np.zeros(num_iters)

    for iter in np.arange(num_iters):
        h = X.dot(theta)
        theta = theta - alpha*(1/m)*(X.T.dot(h-y))
        J_history[iter] = computeCost(X, y, theta)
    return(theta, J_history)
