import numpy as np


def hard_sigmoid(x):
    y = 0.2 * x + 0.5
    return np.clip(y, 0, 1)


def softmax(x, axis=-1):
    y = np.exp(x - np.max(x, axis, keepdims=True))
    return y / np.sum(y, axis, keepdims=True)

# xty modified
def sigmoid(x):
    a = 1. / (1. + np.exp(-x))
    return a


def softplus(x):
    return np.log(1. + np.exp(x))


def softsign(x):
    return x / (1 + np.abs(x))


def tanh(x):
    return np.tanh(x)


def log_logistic(x):
    if x <= 0:
        a = x - np.log(1 + np.exp(x))
    else:
        a = - np.log(1 + np.exp(-x))
    return a
