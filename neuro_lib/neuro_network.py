from random import random

from .activation_function import ActivationFunction
from .neuron import Neuron


class NeuroLayer:
    def __init__(self, input_count: int, neuron_count: int, activation_func: ActivationFunction) -> None:
        self.__activation_func = activation_func
        self.weights = [[random() for _ in range(input_count)] for _ in range(neuron_count)]
        self.offset_weight = [random() for _ in range(neuron_count)]
        self.neurons = tuple(Neuron(activation_func) for _ in range(neuron_count))

    def calculate(self, inputs: list[float]) -> list[float]:
        result = []
        for i, neuron in enumerate(self.neurons):
            S = 0
            if len(inputs) != len(self.weights[i]):
                raise IndexError(f"Length of input not equals length of weights: "
                                 f"{len(inputs)} != {len(self.weights[i])}")
            for j, x in enumerate(inputs):
                S += x * self.weights[i][j]
            S += self.offset_weight[i]
            result.append(neuron.calculate(S))
        return result

    def change_weights(self, x: list[float], y: list[float], reference: list[float], learning_rate: float) -> None:
        for i, neuron_weights in enumerate(self.weights):
            for j in range(len(neuron_weights)):
                neuron_weights[j] = (neuron_weights[j] - learning_rate * x[j] * (y[i] - reference[i])
                                     * self.__activation_func.diff_calculate(self.neurons[i].S))
            self.offset_weight[i] = (self.offset_weight[i] - learning_rate * (y[i] - reference[i])
                                     * self.__activation_func.diff_calculate(self.neurons[i].S))

    def get_weights(self) -> list[list[float]]:
        return [*self.weights, self.offset_weight]


class NeuroNetwork:
    def __init__(self, input_count: int, *layers: NeuroLayer, learning_rate: float | str = 'adaptive') -> None:
        self.adaptive = learning_rate == 'adaptive'
        self.learning_rate = learning_rate
        self.layers = layers

    def learn(self, inputs: list[list[float]], reference: list[list[float]],
              epochs: int = None, error: float = None) -> list[float]:
        if epochs is None and error is None:
            raise ValueError()
        if len(inputs) != len(reference):
            raise ValueError()

        E = []
        if epochs is not None:
            for _ in range(epochs):
                e = self.__learn_step(inputs, reference)
                E.append(e)
                print(f'E: {e: .10f}')
        else:
            while (e := self.__learn_step(inputs, reference)) > error:
                E.append(e)
                print(f'E: {e: .10f}')
            E.append(e)
            print(f'E: {e: .10f}')
        return E

    def predict(self, inputs: list[list[float]]) -> list[list[float]]:
        result = []
        for input_image in inputs:
            X = input_image
            for layer in self.layers:
                X = layer.calculate(X)
            result.append(X)
        return result

    def __learn_step(self, inputs: list[list[float]], reference: list[list[float]]) -> float:
        E = 0
        for i, x in enumerate(inputs):
            y = self.predict([x])[0]
            if self.adaptive:
                self.learning_rate = 1 / (1 + sum(_x ** 2 for _x in x))
            self.layers[-1].change_weights(x, y, reference[i], self.learning_rate)
            E += sum([(_y - _t) ** 2 for _y, _t in zip(y, reference[i])])
        return E / 2
