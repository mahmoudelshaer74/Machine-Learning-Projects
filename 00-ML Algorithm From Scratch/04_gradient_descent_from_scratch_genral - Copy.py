import numpy as np 
from numpy.linalg import norm


def Gradient_Descent(F_Func, intial_vector, Step_Size=0.01, Percsion=0.0001, max_iteration=1000):
    cuur_vector = np.array(intial_vector, dtype=float)
    last_vector = np.array([float('inf')] * len(intial_vector))
    iteration = 0
    vector_list = [cuur_vector]
    
    while norm(cuur_vector - last_vector) > Percsion and iteration < max_iteration:
        last_vector = cuur_vector.copy()
        gradiant = F_Func(cuur_vector)
        cuur_vector -= gradiant * Step_Size
        vector_list.append(cuur_vector.copy())
        iteration += 1
        
    print(f"'The minimum y exists at vector = {cuur_vector}' ")
    return vector_list


def gradient(v):
    x, y = v
    return np.array([
        6 * (x + 2),
        2 * (y - 1)
    ])


def Func():
    def f(x, y, z):
        return np.sin(x) + np.cos(y) + np.sin(z)

    def f_derivdx(x):
        return np.cos(x)

    def f_derivdy(y):
        return -np.sin(y)

    def f_derivdz(z):
        return np.cos(z)

    def fderiv(state):
        deriv = [f_derivdx, f_derivdy, f_derivdz]
        gradiants = []
        for gi, value in zip(deriv, state):
            gradiants.append(gi(value))
        return np.array(gradiants)

    inital_x, inital_y, intial_z = 1, 2, 3.5
    state = np.array([inital_x, inital_y, intial_z])
    mn = Gradient_Descent(fderiv, state)
    final_vector = mn[-1]

    mn_output = f(
        final_vector[0],
        final_vector[1],
        final_vector[2]
    )

    print(
        f'Initial start at {state} '
        f'ends at point: {final_vector} '
        f'with minimum value {mn_output}'
    )


if __name__ == '__main__':
    print("--- Running Gradient Descent Optimization ---")
    Gradient_Descent(gradient, intial_vector=[5, -3], Step_Size=0.1)
    
    print('--------------------------------------------------------------------------------')
    Func()