from random import random

from .activation_function import ActivationFunction
from .neuron import Neuron


class NeuroLayer:
    def __init__(self, input_count: int, neuron_count: int, activation_func: ActivationFunction) -> None:
        self.weights = [[random() for _ in range(input_count)] for _ in range(neuron_count)]
        self.offset_weight = [random() for _ in range(neuron_count)]
        self.neurons = tuple(Neuron(activation_func) for _ in range(neuron_count))

    def calculate(self, inputs: list[float]) -> list[float]:
        result = []
        for i, neuron in enumerate(self.neurons):
            S = 0
            for j, x in enumerate(inputs):
                S += x * self.weights[i][j]
            S += self.offset_weight[i]
            result.append(neuron.calculate(S))
        return result

    def change_weights(self, x: list[float], y: list[float], reference: list[float], learning_rate: float) -> None:
        for i, neuron_weights in enumerate(self.weights):
            for j in range(len(neuron_weights)):
                neuron_weights[j] = neuron_weights[j] - learning_rate * x[j] * (y[i] - reference[i])
            self.offset_weight[i] = self.offset_weight[i] - learning_rate * (y[i] - reference[i])

    def get_weights(self) -> list[list[float]]:
        return [*self.weights, self.offset_weight]


class NeuroNetwork:
    def __init__(self, input_count: int, learning_rate: float, *layers: NeuroLayer) -> None:
        self.learning_rate = learning_rate
        self.layers = layers

    def learn(self, inputs: list[list[float | int]], reference: list[list[float]], epochs: int) -> None:
        if len(inputs) != len(reference):
            return

        for _ in range(epochs):
            for i, x in enumerate(inputs):
                y = self.predict([x])[0]
                self.layers[-1].change_weights(x, y, reference[i], self.learning_rate)

    def predict(self, inputs: list[list[float | int]]) -> list[list[float]]:
        result = []
        for input_image in inputs:
            X = input_image
            for layer in self.layers:
                X = layer.calculate(X)
            result.append(X)
        return result
