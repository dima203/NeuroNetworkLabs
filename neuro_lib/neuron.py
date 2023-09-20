from .activation_function import ActivationFunction


class Neuron:
    def __init__(self, activation_function: ActivationFunction) -> None:
        self.__function = activation_function

    def calculate(self, x: float) -> float:
        return self.__function.calculate(x)
