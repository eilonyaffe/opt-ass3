import numpy as np
from scipy.sparse import random
import scipy.sparse as sparse
n = 256
A = random(n, n, 5 / n, dtype=float)
v = np.random.rand(n)
v = sparse.spdiags(v, 0, v.shape[0], v.shape[0], 'csr')
A = A.transpose() * v * A + 0.1*sparse.eye(n)
b = np.random.randn(n)
sigma = 0.001
A = A.toarray()


def weighted_jacobi_test(A, b, sigma, weight,iter):
    n = len(b)
    diagonal_elements = np.diag(A)
    diagonal_matrix = np.zeros_like(A)
    np.fill_diagonal(diagonal_matrix, diagonal_elements)
    min = [weight, 99]
    while(weight<1):
        x = np.zeros(n)
        for _ in range(iter):
            x_new = x + np.dot(weight*np.linalg.inv(diagonal_matrix),(b-np.dot(A,x)))
            if np.linalg.norm(np.dot(A, x_new) - b) / np.linalg.norm(b) < sigma or np.linalg.norm(
                    x_new - x) / np.linalg.norm(x_new) < sigma:
                if _<min[1]:
                    min[0]=weight
                    min[1]=_
                    print(min)
            else:
                x = x_new
        weight+=0.01
    return min


def weighted_jacobi(A, b, sigma, weight,iter):
    n = len(b)
    x = np.zeros(n)  # initial guess
    diagonal_elements = np.diag(A)
    diagonal_matrix = np.zeros(A.shape)
    np.fill_diagonal(diagonal_matrix, diagonal_elements)
    k=0
    for _ in range(iter):
        x_new = x + np.dot(weight*np.linalg.inv(diagonal_matrix),(b-np.dot(A,x)))
        if np.linalg.norm(np.dot(A,x) - b) / np.linalg.norm(b) < sigma or np.linalg.norm(
                x_new - x) / np.linalg.norm(x_new) < sigma:
            return x_new, _
        else:
            x = x_new
            k=_
    return x,k


def gauss_seidel(A, b, sigma, iter):
    n = len(b)
    x = np.zeros(n)
    diagonal_el = np.diag(A)
    k=0
    for _ in range(iter):
        x_new = np.zeros(n)
        for i in range(n):
            row_i = A[i, :]
            prefix = np.dot(x_new.copy()[:i], row_i.copy()[:i])
            suffix = np.dot(x.copy()[i + 1:], row_i.copy()[i + 1:])
            desired_sum = prefix + suffix
            if diagonal_el[i] == 0:
                return -1
            else:
                x_new[i] = (1 / diagonal_el[i]) * (b[i] - desired_sum)
        if np.linalg.norm(np.dot(A, x_new) - b) / np.linalg.norm(b) < sigma or np.linalg.norm(
                x_new - x) / np.linalg.norm(x_new) < sigma:
            return x_new, _
        else:
            x = x_new
            k=_
    return x,k


def steepest_descent(A, b, sigma, iter):
    n = len(b)
    x = np.zeros(n)
    r = b.copy()-np.dot(A,x)
    k=0
    for _ in range(iter):
        a_rk = np.dot(A, r)
        alpha = np.dot(r,r)/np.dot(r,a_rk)
        new_x = x + alpha*r
        r = r - alpha*a_rk
        if (np.linalg.norm(np.dot(A, new_x) - b) / np.linalg.norm(b)) < sigma or (np.linalg.norm(
                new_x - x) / np.linalg.norm(new_x)) < sigma:
            return new_x, _
        else:
            x = new_x
            k=_
    return x,k


def conjugate_gradient(A, b, sigma, iter):
    n = len(b)
    x = np.zeros(n)
    r = b.copy()-np.dot(A,x)
    p = b.copy()-np.dot(A,x)
    k = 0
    for _ in range(iter):
        a_pk = np.dot(A, p)
        alpha = np.dot(r,r)/np.dot(p,a_pk)
        new_x = x + alpha*p
        new_r = r - alpha*a_pk
        if (np.linalg.norm(np.dot(A, new_x) - b) / np.linalg.norm(b)) < sigma or (np.linalg.norm(
                new_x - x) / np.linalg.norm(new_x)) < sigma:
            return new_x, _
        else:
            beta = np.dot(new_r,new_r)/np.dot(r,r)
            p = new_r + beta*p
            x = new_x
            r = new_r
            k=_
    return x,k


print(weighted_jacobi(A, b, sigma, 0.38, 100)[1])
print(gauss_seidel(A, b, sigma, 100)[1])
print(steepest_descent(A, b, sigma, 100)[1])
print(conjugate_gradient(A, b, sigma, 100)[1])


