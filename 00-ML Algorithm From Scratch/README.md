# Machine Learning Foundations From Scratch 🚀

This repository contains a step-by-step implementation of core Machine Learning mathematical concepts and optimization algorithms from scratch using Python and NumPy. It tracks the evolutionary path from simple single-variable cost calculations to multi-dimensional gradient descent and polynomial regression.

---

## 📁 Repository Structure & File Directory

| File Name | Description | Key Technologies |
| :--- | :--- | :--- |
| **`01_cost_function_from_scratch.py`** | Basic MSE cost function computation. Brute-forces fixed arrays of weights ($M$) and biases ($C$) to find the minimum loss. | Python Loops, NumPy |
| **`02_gradient_descent_from_scratch.py`** | 1D Gradient Descent optimization with convergence history tracking and 2D matplotlib plotting on a basic convex function. | NumPy, Matplotlib |
| **`02_gradient_descent_from_scratch trail2.py`**| 1D Gradient Descent testing on a non-convex polynomial function starting from multiple local and global initialized positions. | NumPy, Matplotlib |
| **`03_gradient_descent_from_scratch_in2_Dim.py`**| 2D Multi-variable Gradient Descent tracking optimal $(x, y)$ coordinates via partial derivatives using Vector norms. | NumPy, Linalg Norm |
| **`04_gradient_descent_from_scratch_genral - Copy.py`**| Generalized $N$-Dimensional Vectorized Gradient Descent handling dynamic states ($3D$ inputs like $x, y, z$). | NumPy, Math Derivs |
| **`05_cost_func_&_gd_library.py`** | Linear Regression cost and gradient dot-product matrix formulations evaluated on manually designed dummy arrays. | Matrix Dot Product |
| **`06_GD_with_linear_Reg.py`** | Matrix-based Linear Regression using full iterative gradient descent trained and evaluated on the Scikit-Learn Diabetes dataset. | Sklearn, MinMaxScaler |
| **`07_Polynomial_Reg.py`** | Non-linear regression boundary mapping using Sklearn's PolynomialFeatures with visual degree capacity evaluations. | Sklearn Linear Models |

---

## 📊 Core Concepts Implemented

1. **Error Evaluation ($MSE$):** Manual mapping of predicting lines vs ground truth labels to measure foundational cost constraints.
2. **Optimization Mechanics:** Moving parameters along the opposite direction of the calculated mathematical derivative vector.
3. **Multi-Variable Convex Tracking:** Upgrading optimization logic to simultaneously adjust weights across multiple dynamic dimensions via matrix manipulation.
4. **Feature Transformation:** Transforming inputs into non-linear higher-degree dimensional metrics to resolve complex feature curves.

---

## 🛠️ Prerequisites & Installation

To run these scripts locally, make sure you have the standard mathematical stack installed in your Python environment:

```bash
pip install numpy matplotlib scikit-learn