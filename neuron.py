import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class Neuron:
    def __init__(self, input_size, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.weights = 2 * np.random.random((input_size, 1)) - 1

    def train(self, training_inputs, training_labels, iterations):
        for iteration in range(iterations):
            output = self.predict(training_inputs)

            # error correction
            error = training_labels - output
            adjustments = np.dot(training_inputs.T, error * sigmoid_derivative(output))
            self.weights += adjustments

    def predict(self, inputs):
        inputs = inputs.astype(float)
        output = sigmoid(np.dot(inputs, self.weights))
        return output
