import pandas as pd
import numpy as np
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class Perceptron:
    def __init__(self, size, learning_rate):
        self.learning_rate = learning_rate

        self.W = np.random.randn(size + 1, 1).T / 20.0

    def train(self, data):
        for sample, label in data:
            sample = self.format_sample(sample)
            inferred_label = self.infer(sample)

            delta_W = self.learning_rate * (label - inferred_label) * sample
            self.W = self.W + delta_W

    def infer(self, sample):
        output = self.W @ sample
        return self.activation_function(output)

    def activation_function(self, value):
        if value > self.W[0, 0]:
            return 1
        else:
            return -1

    def format_sample(self, sample):
        return np.insert(sample, 0, 1)


if __name__ == "__main__":
    training_data = [(np.array([0, 0]), -1),
                     (np.array([0, 1]), 1),
                     (np.array([1, 0]), -1),
                     (np.array([1, 1]), 1)]

    perceptron = Perceptron(2, 0.05)
    perceptron.train(training_data)

    print("done!")
