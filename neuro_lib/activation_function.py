from abc import ABC, abstractmethod


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
