from typing import List

from autodiff.nn import Parameter

class Optimizer:
    def step(self) -> None:
        raise NotImplementedError
    
    def zero_grad(self) -> None:
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, parameters: List[Parameter], lr: float = 0.1) -> None:
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for layer_parameters in self.parameters:
            for parameter in layer_parameters.values():
                parameter -= self.lr * parameter.gradient 

    def zero_grad(self):
        for layer_parameters in self.parameters:
            for parameter in layer_parameters.values():
                parameter.zero_gradient()