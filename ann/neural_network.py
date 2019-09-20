import utils as u
import numpy as np

class Neuron:
    def __init__(self, size, func=u.sigmoid, d_func=u.dsigmoid):
        self.W = np.random.randn(size, 1).T * 0.01
        self.b = np.random.randn(1, 1) * 0.01
        self.func = func
        self.d_func = d_func

    def predict(self, inputs):
        output = self.W @ inputs + self.b
        return self.func(output)

    def update_weights(self, learning_rate, error, inputs):
        self.W += learning_rate * error * inputs
        self.b += learning_rate * error

class NeuralNetwork:
    def __init__(self, blueprint, learning_rate=0.05):
        self.layers = len(blueprint)
        self.neurons_per_layer = blueprint
        self.learning_rate = learning_rate

        self.model = [[Neuron(self.neurons_per_layer[l-1], u.sigmoid, u.dsigmoid)
                       for _ in range(0, self.neurons_per_layer[l])]
                      for l in range(1, self.layers)]
        self.model.insert(0, [])

    def train(self, data, epochs):
        for _ in range(0, epochs):
            for input, correct in data:
                outputs = self.predict(input)
                self.backpropagate(correct, outputs)

    def predict(self, inputs):
        outputs = [[] for _ in range(0, self.layers)]
        outputs[0] = inputs

        for layer in range(1, self.layers):
            previous_layer = outputs[layer-1]
            layer_output = np.zeros(self.neurons_per_layer[layer])
            for neuron in range(0, self.neurons_per_layer[layer]):
                unit = self.model[layer][neuron]
                layer_output[neuron] = unit.predict(previous_layer)
            outputs[layer] = layer_output

        return outputs

    def backpropagate(self, correct, outputs):
        errors = self.calculate_errors(correct, outputs)
        self.update_weights(errors, outputs[:-1])

    def calculate_errors(self, correct, outputs):
        errors = [[0 for neuron in range(0, self.neurons_per_layer[layer])]
                  for layer in range(0, self.layers)]

        # calculate output layer errors
        for i in range(0, len(outputs[-1])):
            o = outputs[-1][i]
            errors[-1] = [u.dsigmoid(o) * (correct[i] - o)]

        # calculate hidden layer errors
        for layer in reversed(range(0, self.layers-1)):
            for neuron in range(0, self.neurons_per_layer[layer]):
                neuron_output = outputs[layer][neuron]

                error_caused = 0
                for i in range(0, self.neurons_per_layer[layer+1]):
                    unit = self.model[layer+1][i]
                    error_caused += unit.W[0][neuron] \
                                    * errors[layer+1][i] \
                                    * unit.d_func(neuron_output)

                errors[layer][neuron] = error_caused

        return errors

    def update_weights(self, errors, outputs):
        for layer in range(1, self.layers):
            for neuron in range(0, self.neurons_per_layer[layer]):
                error = errors[layer][neuron]
                input = outputs[layer-1]
                self.model[layer][neuron].update_weights(self.learning_rate, error, input)

if __name__ == "__main__":
    # ~ XOR
    training_data = [(np.array([0, 0]), [0]),
                     (np.array([0, 1]), [1]),
                     (np.array([1, 0]), [1]),
                     (np.array([1, 1]), [0])]

    nn = NeuralNetwork([2, 2, 1])
    nn.train(training_data, 100000)

    for input, label in training_data:
        print(input, nn.predict(input)[-1])
