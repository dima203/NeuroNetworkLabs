from .activation_function import ActivationFunction


class Neuron:
    def __init__(self, activation_function: ActivationFunction) -> None:
        self.__function = activation_function
        self.S = 0

    def calculate(self, x: float) -> float:
        self.S = x
        return self.__function.calculate(x)
