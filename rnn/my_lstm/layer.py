import numpy as np
from lstm import utils as util


def fc_forward(X, W, b):
    out = X @ W + b
    cache = (W, X)
    return out, cache


def fc_backward(dout, cache):
    W, h = cache

    dW = h.T @ dout
    db = np.sum(dout, axis=0)
    dX = dout @ W.T

    return dX, dW, db


def relu_forward(X):
    out = np.maximum(X, 0)
    cache = X
    return out, cache


def relu_backward(dout, cache):
    dX = dout.copy()
    dX[cache <= 0] = 0
    return dX


def lrelu_forward(X, a=1e-3):
    out = np.maximum(a * X, X)
    cache = (X, a)
    return out, cache


def lrelu_backward(dout, cache):
    X, a = cache
    dX = dout.copy()
    dX[X < 0] *= a
    return dX


def sigmoid_forward(X):
    out = util.sigmoid(X)
    cache = out
    return out, cache


def sigmoid_backward(dout, cache):
    return cache * (1. - cache) * dout


def tanh_forward(X):
    out = np.tanh(X)
    cache = out
    return out, cache


def tanh_backward(dout, cache):
    dX = (1 - cache**2) * dout
    return dX


def dropout_forward(X, p_dropout):
    u = np.random.binomial(1, p_dropout, size=X.shape) / p_dropout
    out = X * u
    cache = u
    return out, cache


def dropout_backward(dout, cache):
    dX = dout * cache
    return dX


def bn_forward(X, gamma, beta, cache, momentum=.9, train=True):
    running_mean, running_var = cache

    if train:
        mu = np.mean(X, axis=0)
        var = np.var(X, axis=0)

        X_norm = (X - mu) / np.sqrt(var + 1e-8)
        out = gamma * X_norm + beta

        cache = (X, X_norm, mu, var, gamma, beta)

        running_mean = util.exp_running_avg(running_mean, mu, momentum)
        running_var = util.exp_running_avg(running_var, var, momentum)
    else:
        X_norm = (X - running_mean) / np.sqrt(running_var + 1e-8)
        out = gamma * X_norm + beta
        cache = None

    return out, cache, running_mean, running_var


def bn_backward(dout, cache):
    X, X_norm, mu, var, gamma, beta = cache

    N, D = X.shape

    X_mu = X - mu
    std_inv = 1. / np.sqrt(var + 1e-8)

    dX_norm = dout * gamma
    dvar = np.sum(dX_norm * X_mu, axis=0) * -.5 * std_inv**3
    dmu = np.sum(dX_norm * -std_inv, axis=0) + dvar * np.mean(-2. * X_mu, axis=0)

    dX = (dX_norm * std_inv) + (dvar * 2 * X_mu / N) + (dmu / N)
    dgamma = np.sum(dout * X_norm, axis=0)
    dbeta = np.sum(dout, axis=0)

    return dX, dgamma, dbeta