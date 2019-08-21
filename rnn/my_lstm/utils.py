import numpy as np


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def exp_running_avg(running, new, gamma=.9):
    return gamma * running + (1. - gamma) * new


def cross_entropy(prediction, truth):
    return - np.log(prediction[truth][0])