import numpy as np
import matplotlib.pyplot as plt


def Gradient_Descent(F_deriv, intial_X, Step_Size=0.001, Precision=0.00001):
    curr_x = intial_X
    last_x = float("inf")

    x_list = [curr_x]

    while abs(curr_x - last_x) > Precision:
        last_x = curr_x
        gradient = F_deriv(curr_x)
        curr_x = curr_x - Step_Size * gradient
        x_list.append(curr_x)

    return curr_x, x_list


def Visualizations(f_func, range_start, range_end, x_list, plt_title):
    X = np.linspace(range_start, range_end, 50)
    y = f_func(X)

    plt.plot(X, y)
    plt.title(plt_title)
    plt.xlabel('x')
    plt.ylabel('y')

    y_points = [f_func(x) for x in x_list]
    plt.scatter(x_list, y_points)
    plt.show()


def trail2():
    def f(x):
        return (x ** 4 - 6 * x ** 2 - x - 1)  # x⁴ - 6x² - x - 1

    def f_derivative(x):
        return 4 * x ** 3 - 12 * x

    func_name = 'Gradient Descent on x⁴ - 6x² - x -1'
    
    for intial_x in [-2.4, -0.15, 0.1, 2.39]:
        title = f'{func_name}: starting from {intial_x}'
        
        minimum_x, history = Gradient_Descent(f_derivative, intial_x, Step_Size=0.001)
        
        print(f'Initial x = {intial_x:.2f} | Found Minimum x = {minimum_x:.4f} | Minimum y = {f(minimum_x):.4f}')
        Visualizations(f, -2.5, 2.5, history, title)


if __name__ == '__main__':
    print("--- Running Gradient Descent Optimization & Visualizations ---")
    trail2()