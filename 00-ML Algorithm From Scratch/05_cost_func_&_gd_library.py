import numpy as np 
from numpy.linalg import norm

def f(X,T,weights):
    n=X.shape[0]
    pred=np.dot(X,weights)
    print(pred)
    error=pred-T
    print(error)
    cost=error.T.dot(error)/(2*n)
    return cost
def f_derivative(X,T,weights):
    n=X.shape[0]
    pred=np.dot(X,weights)
    print(pred)
    error = pred - T
    gradiant=X.T @ error / n
    return gradiant
X = np.array([0, 0.2, 0.4, 0.8, 1.0])
t = 5 + X

X = X.reshape((-1, 1))  
X = np.hstack([np.ones((X.shape[0], 1)), X])  

print(X.shape) 

weights = np.array([0.8, 0.5]) 

print(f(X, t, weights))

print(f_derivative(X, t, weights))  