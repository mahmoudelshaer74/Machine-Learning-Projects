import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import MinMaxScaler

def load_diabets_scaled():
    from sklearn.datasets import load_diabetes
    diabetes=load_diabetes()
    X,target=diabetes.data,diabetes.target
    scaler=MinMaxScaler()
    X=scaler.fit_transform(X)
    return X,target

def cost(X,Weights,Target):
    sample=X.shape[0]
    pred=np.dot(X,Weights)
    error=pred - Target
    cost=error.T.dot(error)/(2*sample)
    return cost

def F_deriv(X,Weights,Target):
    sample=X.shape[0]
    pred=np.dot(X,Weights)
    error=pred - Target
    gradiant=X.T.dot(error)/sample
    return gradiant

def Gradient_Descent_linearRegression(X,T,Step_Size=0.01,precision=0.00001,max_iter=1000):
    examples,features=X.shape
    iter=0
    Curr_weights=np.random.rand(features)
    last_weights=Curr_weights+100*precision
    print(f'Initial Random Cost: {cost(X, Curr_weights,Target)}')
    while norm (last_weights-Curr_weights) > precision and iter < max_iter:
        last_weights=Curr_weights.copy()
        gradiant=F_deriv(X,Curr_weights,Target)
        Curr_weights-=gradiant*Step_Size
        print(f"Iter {iter+1} - Cost: {cost(X, Curr_weights, T)}") 
        iter += 1
    print(f'Total Iterations {iter}')
    print(f'Optimal Cost: {cost(X, Curr_weights, Target)}')
    
    return Curr_weights

X, Target = load_diabets_scaled()
optimal_weights = Gradient_Descent_linearRegression(X, Target)