from .neuro_network import NeuroLayer, Neuron
from .activation_function import BipolarStepFunction, StepFunction

from random import randint


class HopfieldLayer(NeuroLayer):
    def __init__(self, neuron_count: int, activation_function:  StepFunction | BipolarStepFunction) -> None:
        self.__activation_function = activation_function
        self.weights = [[0 for _ in range(neuron_count)] for _ in range(neuron_count)]
        self.neurons = tuple(Neuron(activation_function) for _ in range(neuron_count))

    def calculate(self, inputs: list[int]) -> list[int]:
        input_vector = inputs.copy()
        output_vector = [0 for _ in range(len(input_vector))]
        i_2_vector = [0 for _ in range(len(input_vector))]
        i_1_vector = [0 for _ in range(len(input_vector))]
        while True:
            neuron_index = randint(0, len(self.neurons) - 1)
            for i in range(len(input_vector)):
                S = 0
                for j in range(len(input_vector)):
                    S += self.weights[j][i] * input_vector[j]
                output_vector[i] = self.neurons[i].calculate(S)
            if i_1_vector == output_vector and i_2_vector == input_vector:
                break
            i_2_vector = i_1_vector.copy()
            i_1_vector = input_vector.copy()
            input_vector = output_vector.copy()
        return output_vector

    def init_weights(self, inputs: list[list[int]]) -> None:
        for i in range(len(inputs[0])):
            for j in range(len(inputs[0])):
                value = 0
                for k in range(len(inputs)):
                    if isinstance(self.__activation_function, BipolarStepFunction):
                        value += inputs[k][i] * inputs[k][j]
                    elif isinstance(self.__activation_function, StepFunction):
                        value += (2 * inputs[k][i] - 1) * (2 * inputs[k][j] - 1)
                self.weights[j][i] = value

    def get_weights(self) -> list[list[float]]:
        return self.weights


class HopfieldNetwork:
    def __init__(self, layer: HopfieldLayer) -> None:
        self.layer = layer

    def learn(self, inputs: list[list[int]]) -> None:
        self.layer.init_weights(inputs)

    def predict(self, inputs: list[list[int]]) -> list[list[int]]:
        result = []
        for input_vector in inputs:
            result.append(self.layer.calculate(input_vector))
        return result
