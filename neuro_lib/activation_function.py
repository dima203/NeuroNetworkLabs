from abc import ABC, abstractmethod
from math import exp


class ActivationFunction(ABC):
    @abstractmethod
    def calculate(self, x: float) -> float: ...

    @abstractmethod
    def diff_calculate(self, x: float) -> float: ...


class BipolarStepFunction(ActivationFunction):
    def calculate(self, x: float) -> float:
        if x >= 0:
            return 1
        else:
            return -1

    def diff_calculate(self, x: float) -> float:
        return 1


class StepFunction(ActivationFunction):
    def calculate(self, x: float) -> float:
        if x >= 0:
            return 1
        else:
            return 0

    def diff_calculate(self, x: float) -> float:
        return 1


class LinearFunction(ActivationFunction):
    def calculate(self, x: float) -> float:
        return x

    def diff_calculate(self, x: float) -> float:
        return 1


class SigmoidFunction(ActivationFunction):
    def __init__(self, a: float = 1, b: float = 0):
        self.a = a
        self.b = b

    def calculate(self, x: float) -> float:
        return self.a / (1 + exp(-x)) - self.b

    def diff_calculate(self, x: float) -> float:
        return self.a * exp(-x) / (exp(-x) + 1) ** 2
