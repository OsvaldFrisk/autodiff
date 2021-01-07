from typing import Callable, NamedTuple, Optional, Union, List
import numpy as np


class Adjoint(NamedTuple):
    """partial derivatives of the output with respect to the intermediate variables"""
    tensor: 'Tensor'
    gradient_func: Callable[['Tensor'], 'Tensor']


Tensorable = Union[np.ndarray, list, float]


def ensure_tensor(tensorable: Tensorable):
    if isinstance(tensorable, np.ndarray):
        return tensorable
    else:
        return np.array(tensorable)


class Tensor:
    def __init__(self,
                 data: Tensorable,
                 requires_gradient: bool = False,
                 depends_on: List[Adjoint] = None) -> None:
        self._data = ensure_tensor(data)
        self.requires_gradient = requires_gradient
        self.depends_on = depends_on or []
        self.shape = self._data.shape
        self.gradient: Optional['Tensor'] = None

        if self.requires_gradient:
            self.zero_gradient()

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, new_data: np.ndarray) -> None:
        self._data = new_data
        # setting data invalidates gradient
        self.gradient = None

    def zero_gradient(self) -> None:
        self.gradient = Tensor(np.zeros(self.shape, dtype=float))

    def backward(self, gradient: 'Tensor' = None) -> None:
        assert self.requires_gradient, "backward was called on a tensor that does not require a gradient"

        if gradient is None:
            if self.shape == ():
                gradient = Tensor(1.)
            else:
                raise RuntimeError("No gradient specified")
        
        self.gradient.data = self.gradient.data + gradient.data  # type: ignore

        for dependency in self.depends_on:
            backward_gradient = dependency.gradient_func(gradient.data)
            dependency.tensor.backward(Tensor(backward_gradient))

    def sum(self) -> 'Tensor':
        return _sum(self)


# Tensor operations
def _sum(tensor: 'Tensor') -> 'Tensor':
    """
    Inputs a tensor to sum, returns 0-tensor sum of elements.
    """
    data = tensor.data.sum()
    requires_gradient = tensor.requires_gradient
    depends_on: List[Adjoint] = []

    if requires_gradient:

        def gradient_func_sum(gradient: 'Tensor') -> 'Tensor':
            return gradient * np.ones(tensor.shape).data

        depends_on.append(Adjoint(tensor, gradient_func_sum))

    return Tensor(data, requires_gradient, depends_on)
