import numpy as np
import math

def sigmoid(x):
    x = np.clip(x, -500, 500)
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

def one_hot(size, peak):
    output = np.zeros(size)
    output[peak] = 1
    return output

def gaussian_prob(mean, var, x):
    eps = 1e-4
    coeff = 1.0 / math.sqrt(2.0 * math.pi * var + eps)
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * var + eps)))
    return coeff * exponent

def pd_kfold_iteration_i(dataset, i, k):
    fs = len(dataset) // k
    validate_data = dataset.iloc[i*fs : (i+1)*fs]
    training_data = dataset.iloc[0*fs : i*fs].append( \
        dataset.iloc[(i+1)*fs :])
    return training_data, validate_data

def f1_score(precision, recall):
    return (2 * precision * recall) / (precision + recall)