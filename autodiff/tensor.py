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

    def zero_gradient(self) -> None:
        self.gradient = Tensor(np.zeros(self.shape, dtype=float))