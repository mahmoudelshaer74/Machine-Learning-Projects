import numpy as np
import matplotlib.pyplot as plt


def Gradient_Descent(F_deriv, intial_X,
                     Step_Size=0.001,
                     Precision=0.00001):

    curr_x = intial_X
    last_x = float("inf")

    x_list = [curr_x]

    while abs(curr_x - last_x) > Precision:

        last_x = curr_x

        gradient = F_deriv(curr_x)

        curr_x = curr_x - Step_Size * gradient

        x_list.append(curr_x)

    return curr_x, x_list


def Visualizations(f_func,
                   range_start,
                   range_end,
                   x_list,
                   plt_title):

    X = np.linspace(range_start, range_end, 50)

    y = f_func(X)

    plt.plot(X, y)

    plt.title(plt_title)

    plt.xlabel('x')

    plt.ylabel('y')

    y_points = [f_func(x) for x in x_list]

    plt.scatter(x_list, y_points)

    plt.show()


def trial1():

    def f(x):
        return 3 * x**2 + 4 * x + 7

    def f_derivative(x):
        return 6 * x + 4

    func_name = 'Gradient Descent on 3x^2 + 4x + 7'

    for inital_x in [-7.5, 5, -2/3]:

        minimum_x, history = Gradient_Descent(
            F_deriv=f_derivative,
            intial_X=inital_x
        )

        print(f'Initial x = {inital_x}')
        print(f'Minimum x = {minimum_x}')
        print(f'Minimum y = {f(minimum_x)}')

        Visualizations(f, -10, 10, history, func_name)


if __name__ == '__main__':

    trial1()