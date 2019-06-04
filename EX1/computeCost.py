import numpy as np

def computeCost(X, y, theta=[[0],[0]]):
    m = y.size
    J = 0

    h = X.dot(theta)

    J = 1/(2*m)*np.sum(np.square(h-y))

    return(J)

