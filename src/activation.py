import numpy as np


def relu(x):
    return np.maximum(0, x)


def leaky_relu(x, slope):
    return np.maximum(slope, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)
