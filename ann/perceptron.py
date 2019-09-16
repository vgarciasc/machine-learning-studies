import pandas as pd
import numpy as np
import math

from enum import Enum


class PerceptronTraining(Enum):
    PERCEPTRON_RULE = 1
    GRADIENT_DESCENT = 2
    STOCHASTIC_GRADIENT_DESCENT = 3


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class Perceptron:
    def __init__(self, size, learning_rate):
        self.learning_rate = learning_rate

        self.W = np.random.randn(size, 1).T * 0.01
        self.b = np.random.randn(1, 1) * 0.01

    def train(self, data, epochs, training_style=PerceptronTraining.PERCEPTRON_RULE):
        if training_style == PerceptronTraining.PERCEPTRON_RULE:
            self.perceptron_rule_train(data, epochs)
        elif training_style == PerceptronTraining.GRADIENT_DESCENT:
            self.gradient_descent_train(data, epochs)
        elif training_style == PerceptronTraining.STOCHASTIC_GRADIENT_DESCENT:
            self.stochastic_gradient_descent_train(data, epochs)

    def perceptron_rule_train(self, data, epochs):
        for _ in range(0, epochs):
            for input, label in data:
                prediction = self.predict(input)

                self.W += self.learning_rate * (label - prediction) * input
                self.b += self.learning_rate * (label - prediction)

    def gradient_descent_train(self, data, epochs):
        for _ in range(0, epochs):
            delta_W = np.zeros_like(self.W)
            delta_b = 0

            for input, label in data:
                prediction = self.predict(input)

                delta_W += delta_W + self.learning_rate * (label - prediction) * input
                delta_b += delta_b + self.learning_rate * (label - prediction)

            self.W += delta_W
            self.b += delta_b

    def stochastic_gradient_descent_train(self, data, epochs):
        for _ in range(0, epochs):
            for input, label in data:
                prediction = self.predict(input)

                self.W += self.learning_rate * (label - prediction) * input
                self.b += self.learning_rate * (label - prediction)

    def predict(self, input, use_activ_func=True):
        output = self.W @ input + self.b
        return self.activation_function(output) if use_activ_func else output

    def activation_function(self, value):
        return 1 if value > 0 else -1


if __name__ == "__main__":
    accuracies = []
    for i in range(0, 10):
        training_data = [(np.array([0, 0]), -1),
                         (np.array([0, 1]), 1),
                         (np.array([1, 0]), 1),
                         (np.array([1, 1]), -1)]

        perceptron = Perceptron(2, 0.05)
        perceptron.train(training_data, 100, PerceptronTraining.STOCHASTIC_GRADIENT_DESCENT)

        correct = 0
        for sample, label in training_data:
            correct = correct + (1 if perceptron.predict(sample) == label else 0)

        accuracies.append(correct / len(training_data))

    print("accuracy: %d!", np.mean(accuracies))
