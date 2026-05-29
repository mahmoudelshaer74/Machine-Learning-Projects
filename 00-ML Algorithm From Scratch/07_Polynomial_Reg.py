import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def Generate_data(n=200):
    x=np.random.uniform(-20,20,n)
    y = 5 + x + 4 * x ** 2 + 5 * x ** 3 - 8 * x ** 4
    scale=MinMaxScaler()
    x=scale.fit_transform(x.reshape(-1,1)).reshape(-1)
    mu, sigma = 0, 0.02
    noise = np.random.normal(mu, sigma, n)
    y=scale.fit_transform(y.reshape(-1,1)).reshape(-1) + noise
    return x,y

def visualize2D(x, y, is_scatter=True, tite='x vs y'):
    if is_scatter:
        plt.scatter(x, y)
    else:
        plt.plot(x, y)
    plt.title(tite)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()

def visualize_polynomial(model, degree):
    from sklearn.preprocessing import PolynomialFeatures
    n=500
    x=np.random.uniform(-50,50,n)
    poly=PolynomialFeatures(degree=degree,include_bias=True)
    x_new=poly.fit_transform(x.reshape(-1,1))
    t=model.predict(x_new)
    visualize2D(x,t)

def learn_polynomial(x,t,degree=1):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import PolynomialFeatures
    poly=PolynomialFeatures(degree=degree,include_bias=True)
    x_new=poly.fit_transform(x.reshape(-1,1))
    ## ----------------------
    model=LinearRegression()
    model.fit(x_new,t)
    pred_t=model.predict(x_new)
    error=mean_squared_error(t,pred_t)
    return error , model

def try_polynomials():
    x,t=Generate_data()
    visualize2D(x,t)
    degrees,errors=[],[]
    for degree in range(1,9):
        error,model=learn_polynomial(x,t,degree)
        print(f'Degree{degree} has error {error} {(abs(model.coef_)).sum()}- {model.coef_}')
        visualize_polynomial(model, degree)
        degrees.append(degree)
        errors.append(error)
    visualize2D(degrees, errors, is_scatter=False, tite='Degree vs Error')  

try_polynomials()