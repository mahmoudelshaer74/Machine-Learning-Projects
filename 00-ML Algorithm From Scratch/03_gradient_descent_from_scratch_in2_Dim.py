import numpy as np 
from numpy.linalg import norm
def gradient_descent(fderiv_dx,fderiv_dy,inital_x,inital_y,step_size=0.001,percision=0.00001,max_iter=1000):
    curr_vector=np.array([inital_x,inital_y])
    last_xy=np.array([float('inf'),float('inf')])
    list_xy=[curr_vector]
    itera = 0
    while norm(curr_vector-last_xy) > percision and itera < max_iter:
        print(curr_vector)
        last_xy=curr_vector.copy()
        gx=fderiv_dx(curr_vector[0],curr_vector[1])
        gy=fderiv_dy(curr_vector[0],curr_vector[1])
        gradient = np.array([gx, gy])
        curr_vector-=gradient*step_size
        list_xy.append(curr_vector.copy())
        itera+=1
    print(f'The minimum z exists at (x,y) = {curr_vector}')
    return list_xy

def trial1():
    def f(x, y):
        return 3 * (x + 2) ** 2 + (y - 1) ** 2       # 3(x + 2)² + (y - 1)²

    # https://calculator-derivative.com/partial-derivative-calculator
    def fderiv_dx(x, y):
        return 6 * (x + 2)

    def fderiv_dy(x, y):
        return 2 * (y - 1)

    func_name = 'Gradient Descent on 2x² - 4x y + y⁴ + 2'

    inital_x, inital_y = -5.0, 2.5
    list_xy = gradient_descent(fderiv_dx, fderiv_dy, inital_x, inital_y)
    # The minimum z exists at (x,y) = [-2.00730307  1.20259678]

if __name__ == '__main__':
    print("--- Running Gradient Descent Optimization  ---")
    trial1()