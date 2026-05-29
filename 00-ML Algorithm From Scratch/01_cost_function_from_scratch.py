import numpy as np

M = [2, 5, 6, 4, 8, 7]
C = [1, 3, 6]

X = [1, 2, 3, 4, 5, 6]
Y = [6, 5, 4, 3, 2, 1]


def compute_cost(m, c):

    cost = 0
    n = len(Y)

    for x, y_gt in zip(X, Y):

        y_pred = m * x + c

        error = y_gt - y_pred

        squared_error = error ** 2

        cost += squared_error

    return cost / (n * 2)


if __name__ == '__main__':

    print("Cost =", compute_cost(2, 6))

    best_cost = float('inf')

    for m in M:

        for c in C:

            this_cost = compute_cost(m, c)

            if best_cost > this_cost:

                best_cost = this_cost

                best_m = m

                best_c = c

    print(f'Best m = {best_m}')
    print(f'Best c = {best_c}')
    print(f'Best cost = {best_cost}')