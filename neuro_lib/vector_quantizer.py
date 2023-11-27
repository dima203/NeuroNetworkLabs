from .neuro_network import NeuroLayer
from .activation_function import StepFunction


class QuantizerLayer(NeuroLayer):
    def __init__(self, input_count: int, neuron_count: int, activation_func: StepFunction) -> None:
        super().__init__(input_count, neuron_count, activation_func)

    def change_weights(self, x: list[float], errors: list[float], learning_rate: float) -> list[float]:
        d = []
        for neuron_weights in self.weights:
            _d = 0
            for _x, weight in zip(x, neuron_weights):
                _d += (_x - weight) ** 2
            d.append(_d ** 0.5)
        k = d.index(min(d))
        for i in range(len(self.weights[k])):
            self.weights[k][i] += learning_rate * (x[i] - self.weights[k][i])

    def calculate(self, inputs: list[float]) -> list[float]:
        d = []
        result = [0] * len(self.neurons)
        for neuron_weights in self.weights:
            _d = 0
            for _x, weight in zip(inputs, neuron_weights):
                _d += (_x - weight) ** 2
            d.append(_d ** 0.5)
        k = d.index(min(d))
        result[k] = 1
        return result


class VectorQuantizer:
    def __init__(self, layer: QuantizerLayer) -> None:
        self.layer = layer

    def learn(self, inputs: list[list[float]], epochs: int) -> None:
        for epoch in range(epochs):
            learning_rate = 1 / (epoch + 1)
            for input_vector in inputs:
                self.layer.change_weights(input_vector, [0], learning_rate)

    def predict(self, inputs: list[list[float]]) -> list[list[float]]:
        result = []
        for input_vector in inputs:
            result.append(self.layer.calculate(input_vector))
        return result
