import numpy as np


def jacobi(A, b, sigma):
    n = len(b)
    x = np.zeros(n)  # initial guess
    diagonal_el = np.diag(A)
    flag = True
    k = 1  # TODO delete later
    while flag:
        x_new = np.zeros(n)
        for i in range(n):
            row_i = A[i, :].copy()
            row_i[i] = 0
            desired_sum = np.dot(row_i, x)
            if diagonal_el[i] == 0:  # generally assuming there are no zero entries in the diagonal TODO make sure
                return -1
            else:
                x_new[i] = (1 / diagonal_el[i]) * (b[i] - desired_sum)
        if np.linalg.norm(np.dot(A, x_new) - b) / np.linalg.norm(b) < sigma or np.linalg.norm(
                x_new - x) / np.linalg.norm(x_new) < sigma:
            return x_new, k
        else:
            x = x_new
            k += 1
    return -1


def gauss_seidel(A, b, sigma):
    n = len(b)
    x = np.zeros(n)
    diagonal_el = np.diag(A)
    flag = True
    k = 1  # TODO delete later
    while flag:
        x_new = np.zeros(n)
        for i in range(n):
            row_i = A[i, :]
            prefix = np.dot(x_new.copy()[:i], row_i.copy()[:i])
            suffix = np.dot(x.copy()[i + 1:], row_i.copy()[i + 1:])
            desired_sum = prefix + suffix
            if diagonal_el[i] == 0:  # generally assuming there are no zero entries in the diagonal TODO make sure
                return -1
            else:
                x_new[i] = (1 / diagonal_el[i]) * (b[i] - desired_sum)
        if np.linalg.norm(np.dot(A, x_new) - b) / np.linalg.norm(b) < sigma or np.linalg.norm(
                x_new - x) / np.linalg.norm(x_new) < sigma:
            return x_new, k
        else:
            x = x_new
            k += 1
    return -1


def steepest_descent(A, b, sigma):
    n = len(b)
    x = np.zeros(n)  # Initial guess
    r = b.copy()-np.dot(A,x)  # Initial residual
    k = 0  # Iteration counter

    while True:
        a_rk = np.dot(A, r)
        alpha = np.dot(r,r)/np.dot(r,a_rk)
        new_x = x + alpha*r
        r = r - alpha*a_rk
        if (np.linalg.norm(np.dot(A, new_x) - b) / np.linalg.norm(b)) < sigma or (np.linalg.norm(
                new_x - x) / np.linalg.norm(new_x)) < sigma:
            return new_x, k
        else:
            x = new_x
            k += 1

def conjugate_gradient(A, b, sigma):
    n = len(b)
    x = np.zeros(n)  # Initial guess
    r = b.copy()-np.dot(A,x)  # Initial residual
    p = b.copy()-np.dot(A,x)  # Initial residual
    k = 0  # Iteration counter

    while True:
        a_pk = np.dot(A, p)
        alpha = np.dot(r,r)/np.dot(p,a_pk)
        new_x = x + alpha*p
        new_r = r - alpha*a_pk
        if (np.linalg.norm(np.dot(A, new_x) - b) / np.linalg.norm(b)) < sigma or (np.linalg.norm(
                new_x - x) / np.linalg.norm(new_x)) < sigma:
            return new_x, k
        else:
            beta = np.dot(new_r,new_r)/np.dot(r,r)
            p = new_r + beta*p
            x = new_x
            r = new_r
            k += 1



A = np.array([[4, -1, 1],
              [4, -8, 1],
              [-2, 1, 5]])
b = np.array([7, -21, 15])
sigma = 0.00001
A_spd = np.array([[2, -1, 0],
              [-1, 2, -1],
              [0, -1, 2]])


# non SPD
print(jacobi(A, b, sigma))
print(gauss_seidel(A, b, sigma))
print(steepest_descent(A_spd, b, sigma))

# SPD
print(jacobi(A_spd, b, sigma))
print(gauss_seidel(A_spd, b, sigma))
print(steepest_descent(A_spd, b, sigma))
print(conjugate_gradient(A_spd, b, sigma))
