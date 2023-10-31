from random import random

from .activation_function import ActivationFunction
from .neuron import Neuron


class NeuroLayer:
    def __init__(self, input_count: int, neuron_count: int, activation_func: ActivationFunction) -> None:
        self.__activation_func = activation_func
        self.weights = [[random() for _ in range(input_count)] for _ in range(neuron_count)]
        self.offset_weight = [random() for _ in range(neuron_count)]
        self.neurons = tuple(Neuron(activation_func) for _ in range(neuron_count))
        self.outputs = []

    def calculate(self, inputs: list[float]) -> list[float]:
        self.outputs = []
        for i, neuron in enumerate(self.neurons):
            S = 0
            if len(inputs) != len(self.weights[i]):
                raise IndexError(f"Length of input not equals length of weights: "
                                 f"{len(inputs)} != {len(self.weights[i])}")
            for j, x in enumerate(inputs):
                S += x * self.weights[i][j]
            S += self.offset_weight[i]
            self.outputs.append(neuron.calculate(S))
        return self.outputs

    def change_weights(self, x: list[float], errors: list[float], learning_rate: float) -> list[float]:
        result_errors = [0 for _ in range(len(self.weights[0]))]
        for i, neuron_weights in enumerate(self.weights):
            for j in range(len(neuron_weights)):
                result_errors[j] = errors[i] * self.__activation_func.diff_calculate(self.neurons[i].S) * neuron_weights[j]
                neuron_weights[j] = (neuron_weights[j] - learning_rate * x[j] * errors[i]
                                     * self.__activation_func.diff_calculate(self.neurons[i].S))
            self.offset_weight[i] = (self.offset_weight[i] - learning_rate * errors[i]
                                     * self.__activation_func.diff_calculate(self.neurons[i].S))
        return result_errors

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
            errors = [y - t for y, t in zip(y, reference[i])]
            for j in range(len(self.layers) - 1, -1, -1):
                if j != 0:
                    X = self.layers[j - 1].outputs
                else:
                    X = x
                errors = self.layers[j].change_weights(X, errors, self.learning_rate)
            E += sum([(_y - _t) ** 2 for _y, _t in zip(y, reference[i])])
        return E / 2
