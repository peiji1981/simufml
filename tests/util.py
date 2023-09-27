import numpy as np

def approx_eq(value, expect_value, tol):
    return expect_value-tol <= value <= expect_value+tol

def err_ratio(x, y):
    return np.abs((x-y) / y)