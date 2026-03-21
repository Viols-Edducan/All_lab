import numpy as np


def factorial(n, stop=1):
    if n == 0:
        return 1
    elif n == 1:
        return 1
    else:
        if n == stop:
            return n
        else:
            return n * factorial(n - 1, stop)

def c_(n, i):
    if i == 0:
        return 1
    else:
        return int(round(factorial(n, n - i + 1) / factorial(i)))


def bezier_rang_matrix(rang):
    if rang < 2:
        raise ValueError('too low rang')
    power = rang - 1
    return np.array([[(-1) ** (i + j) * c_(power, j) * c_(power - j, i - j) if i >=j else 0
                      for i in range(rang)]
                     for j in range(rang)])

def vector_mul_2d(a, b, c):
    return np.linalg.det(np.array([[*a, 1], [*b, 1], [*c, 1]]))


def symbol(val):
    if val == 0:
        return 0
    return 1 if val > 0 else -1