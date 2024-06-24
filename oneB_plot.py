import numpy as np
from scipy.sparse import random
import scipy.sparse as sparse
import matplotlib.pyplot as plt


n = 256
A = random(n, n, 5 / n, dtype=float)
v = np.random.rand(n)
v = sparse.spdiags(v, 0, v.shape[0], v.shape[0], 'csr')
A = A.transpose() * v * A + 0.1*sparse.eye(n)
b = np.random.randn(n)
sigma = 0.001
weight = 0.38
max_iter = 100
A = A.toarray()


def weighted_jacobi(A, b, sigma, weight, max_iter):
    n = len(b)
    x = np.zeros(n)
    diagonal_elements = np.diag(A)
    diagonal_matrix = np.zeros(A.shape)
    np.fill_diagonal(diagonal_matrix, diagonal_elements)
    norm_residuals = []  # To store norms of residuals at each iteration
    norm_residuals_divided = []

    for k in range(max_iter):
        x_new = x + np.dot(weight*np.linalg.inv(diagonal_matrix),(b-np.dot(A,x)))
        residual_norm = np.linalg.norm(np.dot(A, x_new) - b)
        norm_residuals.append(residual_norm)
        residual_norm_divided = residual_norm/np.linalg.norm(np.dot(A, x) - b)
        norm_residuals_divided.append(residual_norm_divided)

        if residual_norm / np.linalg.norm(b) < sigma or np.linalg.norm(x_new - x) / np.linalg.norm(x_new) < sigma:
            return x_new, k + 1, norm_residuals, norm_residuals_divided
        else:
            x = x_new

    return x, max_iter, norm_residuals, norm_residuals_divided

def gauss_seidel(A, b, sigma, max_iter):
    n = len(b)
    x = np.zeros(n)
    diagonal_elements = np.diag(A)
    norm_residuals = [] 
    norm_residuals_divided = []

    for k in range(max_iter):
        x_new = np.zeros(n)
        for i in range(n):
            row_i = A[i, :]
            prefix = np.dot(x_new[:i], row_i[:i])
            suffix = np.dot(x[i + 1:], row_i[i + 1:])
            desired_sum = prefix + suffix

            if diagonal_elements[i] == 0:
                return -1
            else:
                x_new[i] = (1 / diagonal_elements[i]) * (b[i] - desired_sum)

        residual_norm = np.linalg.norm(np.dot(A, x_new) - b)
        norm_residuals.append(residual_norm)
        residual_norm_divided = residual_norm/np.linalg.norm(np.dot(A, x) - b)
        norm_residuals_divided.append(residual_norm_divided)

        if residual_norm / np.linalg.norm(b) < sigma or np.linalg.norm(x_new - x) / np.linalg.norm(x_new) < sigma:
            return x_new, k + 1, norm_residuals, norm_residuals_divided
        else:
            x = x_new

    return x, max_iter, norm_residuals,norm_residuals_divided

def steepest_descent(A, b, sigma, max_iter):
    n = len(b)
    x = np.zeros(n)
    r = b - np.dot(A, x)
    norm_residuals = [] 
    norm_residuals_divided = []

    for k in range(max_iter):
        a_rk = np.dot(A, r)
        alpha = np.dot(r, r) / np.dot(r, a_rk)
        new_x = x + alpha * r
        r = r - alpha * a_rk

        residual_norm = np.linalg.norm(np.dot(A, new_x) - b)
        norm_residuals.append(residual_norm)
        residual_norm_divided = residual_norm/np.linalg.norm(np.dot(A, x) - b)
        norm_residuals_divided.append(residual_norm_divided)

        if residual_norm / np.linalg.norm(b) < sigma or np.linalg.norm(new_x - x) / np.linalg.norm(new_x) < sigma:
            return new_x, k + 1, norm_residuals, norm_residuals_divided
        else:
            x = new_x

    return x, max_iter, norm_residuals,norm_residuals_divided

def conjugate_gradient(A, b, sigma, max_iter):
    n = len(b)
    x = np.zeros(n)
    r = b - np.dot(A, x)
    p = r.copy()
    norm_residuals = []  
    norm_residuals_divided = []

    for k in range(max_iter):
        a_pk = np.dot(A, p)
        alpha = np.dot(r, r) / np.dot(p, a_pk)
        new_x = x + alpha * p
        new_r = r - alpha * a_pk

        residual_norm = np.linalg.norm(np.dot(A, new_x) - b)
        norm_residuals.append(residual_norm)
        residual_norm_divided = np.linalg.norm(np.dot(A, new_x) - b)/np.linalg.norm(np.dot(A, x) - b)
        norm_residuals_divided.append(residual_norm_divided)

        if residual_norm / np.linalg.norm(b) < sigma or np.linalg.norm(new_x - x) / np.linalg.norm(new_x) < sigma:
            return new_x, k + 1, norm_residuals, norm_residuals_divided
        else:
            beta = np.dot(new_r, new_r) / np.dot(r, r)
            p = new_r + beta * p
            x = new_x
            r = new_r

    return x, max_iter, norm_residuals, norm_residuals_divided

print(weighted_jacobi(A, b, sigma, weight, max_iter)[1])
print(gauss_seidel(A, b, sigma, max_iter)[1])
print(steepest_descent(A, b, sigma, max_iter)[1])
print(conjugate_gradient(A, b, sigma, max_iter)[1])


#
# # Plotting jacobi
# result, iterations, norm_residuals, norm_residuals_divided = weighted_jacobi(A, b, sigma, weight, max_iter)
# plt.figure()
# plt.semilogy(range(1, iterations + 1), norm_residuals, marker='o', linestyle='-')
# plt.xlabel('Iteration $k$')
# plt.ylabel('Logarithm of Residual Norm')
# plt.title('Convergence of Weighted Jacobi Method with w=0.38')
# plt.grid(True)
# plt.show()
#
# plt.figure()
# plt.semilogy(range(1, iterations + 1), norm_residuals_divided, marker='s', linestyle='-')
# plt.xlabel('Iteration $k$')
# plt.ylabel('Logarithm of Residual Norm')
# plt.title('Convergence of Weighted Jacobi Method with w=0.38 (Divided By Norm)')
# plt.grid(True)
# plt.show()
#
#
#
# # Plotting GS
# result, iterations, norm_residuals, norm_residuals_divided = gauss_seidel(A, b, sigma, max_iter)
# plt.figure()
# plt.semilogy(range(1, iterations + 1), norm_residuals, marker='o', linestyle='-')
# plt.xlabel('Iteration $k$')
# plt.ylabel('Logarithm of Residual Norm')
# plt.title('Convergence of Gauss-Seidel Method')
# plt.grid(True)
# plt.show()
#
# # Plotting norm_residuals_divided
# plt.figure()
# plt.semilogy(range(1, iterations + 1), norm_residuals_divided, marker='s', linestyle='-')
# plt.xlabel('Iteration $k$')
# plt.ylabel('Logarithm of Residual Norm')
# plt.title('Convergence of Gauss-Seidel Method (Divided By Norm)')
# plt.grid(True)
# plt.show()
#
# # Plotting SD
# result, iterations, norm_residuals, norm_residuals_divided = steepest_descent(A, b, sigma, max_iter)
# plt.figure()
# plt.semilogy(range(1, iterations + 1), norm_residuals, marker='o', linestyle='-')
# plt.xlabel('Iteration $k$')
# plt.ylabel('Logarithm of Residual Norm')
# plt.title('Convergence of Steepest Descent Method')
# plt.grid(True)
# plt.show()
#
# plt.figure()
# plt.semilogy(range(1, iterations + 1), norm_residuals_divided, marker='s', linestyle='-')
# plt.xlabel('Iteration $k$')
# plt.ylabel('Logarithm of Residual Norm')
# plt.title('Convergence of Steepest Descent Method (Divided By Norm)')
# plt.grid(True)
# plt.show()
#
# # Plotting CG
# result, iterations, norm_residuals, norm_residuals_divided = conjugate_gradient(A, b, sigma, max_iter)
# plt.figure()
# plt.semilogy(range(1, iterations + 1), norm_residuals, marker='o', linestyle='-')
# plt.xlabel('Iteration $k$')
# plt.ylabel('Logarithm of Residual Norm')
# plt.title('Convergence of Conjugate Gradient Method')
# plt.grid(True)
# plt.show()
#
# plt.figure()
# plt.semilogy(range(1, iterations + 1), norm_residuals_divided, marker='s', linestyle='-')
# plt.xlabel('Iteration $k$')
# plt.ylabel('Logarithm of Residual Norm')
# plt.title('Convergence of Conjugate Gradient Method (Divided By Norm)')
# plt.grid(True)
# plt.show()
