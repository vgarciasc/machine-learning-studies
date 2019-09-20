import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def dsigmoid(x):
    return x * (1 - x)

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1 - x**2

def relu(x):
    return x if x > 0 else 0

def drelu(x):
    return 1 if x > 0 else 0