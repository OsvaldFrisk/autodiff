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

    @property
    def T(self):
        return Tensor(self._data.T, self.requires_gradient, self.depends_on)

    @data.setter  # type: ignore
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

    def _ensure_tensor(self, tensorable: Union['Tensor', np.ndarray, list, float]) -> 'Tensor':
        """checks if tensorable is a Tensor, and makes it to Tensor if it is not"""
        if isinstance(tensorable, Tensor):
            return tensorable
        else:
            return Tensor(tensorable)

    def reshape(self, *shape) -> 'Tensor':
        self._data = self._data.reshape(*shape)
        self.shape = self._data.shape
        return self

    def sum(self) -> 'Tensor':
        return _sum(self)

    def __repr__(self) -> str:
        return f"Tensor={self._data}, requires_gradient={self.requires_gradient}"

    def __add__(self, other: Tensorable) -> 'Tensor':
        """called when tensor + other"""
        return _add(self, self._ensure_tensor(other))

    def __radd__(self, other: Tensorable) -> 'Tensor':
        """called when other + tensor"""
        return _add(self._ensure_tensor(other), self)

    def __iadd__(self, other: Tensorable) -> 'Tensor':
        """called when tensor += other"""
        self.data = self.data + self._ensure_tensor(other).data  # type: ignore
        return self

    def __mul__(self, other: Tensorable) -> 'Tensor':
        """called when tensor * other"""
        return _mul(self, self._ensure_tensor(other))

    def __rmul__(self, other: Tensorable) -> 'Tensor':
        """called when other * tensor"""
        return _mul(self._ensure_tensor(other), self)

    def __imul__(self, other: Tensorable) -> 'Tensor':
        """called when tensor *= other"""
        self.data = self.data * self._ensure_tensor(other).data  # type: ignore
        return self

    def __neg__(self) -> 'Tensor':
        return _neg(self)

    def __sub__(self, other: Tensorable) -> 'Tensor':
        """called when tensor - other"""
        return _sub(self, self._ensure_tensor(other))

    def __rsub__(self, other: Tensorable) -> 'Tensor':
        """called when other - tensor"""
        return _sub(self._ensure_tensor(other), self)

    def __isub__(self, other: Tensorable) -> 'Tensor':
        """called when tensor -= other"""
        self.data = self.data - self._ensure_tensor(other).data  # type: ignore
        return self

    def __matmul__(self, other: Tensorable) -> 'Tensor':
        return _matmul(self, self._ensure_tensor(other))

    def __getitem__(self, idxs: Union[List[int], int]) -> 'Tensor':
        return _slice(self, idxs)

    def __truediv__(self, other: Tensorable) -> 'Tensor':
        return _div(self, self._ensure_tensor(other))

    # def __pow__(self, other: Tensorable) -> 'Tensor':
    #     return _pow(self, self._ensure_tensor(other))

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

def _add(left_tensor: 'Tensor', right_tensor: 'Tensor') -> 'Tensor':
    """
    Returns the sum of the two tensors
    """
    data = left_tensor.data + right_tensor.data
    requires_gradient = left_tensor.requires_gradient or right_tensor.requires_gradient
    depends_on: List[Adjoint] = []

    if left_tensor.requires_gradient:
        def gradient_func_add_left(gradient: np.ndarray) -> np.ndarray:
            ndims_added = gradient.ndim - left_tensor.data.ndim

            # Added dimensions are summed out
            for _ in range(ndims_added):
                gradient = gradient.sum(axis=0)

            # In case of broadcasting we don't want to remove dims
            for idx, dim in enumerate(left_tensor.shape):
                if dim == 1:
                    gradient = gradient.sum(axis=idx, keepdims=True)

            return gradient

        depends_on.append(Adjoint(left_tensor, gradient_func_add_left))

    if right_tensor.requires_gradient:
        def gradient_func_add_right(gradient: np.ndarray) -> np.ndarray:
            ndims_added = gradient.ndim - right_tensor.data.ndim

            for _ in range(ndims_added):
                gradient = gradient.sum(axis=0)

            for idx, dim in enumerate(right_tensor.shape):
                if dim == 1:
                    gradient = gradient.sum(axis=idx, keepdims=True)

            return gradient

        depends_on.append(Adjoint(right_tensor, gradient_func_add_right))

    return Tensor(data, requires_gradient, depends_on)

def _mul(left_tensor: 'Tensor', right_tensor: 'Tensor') -> 'Tensor':
    """
    Returns the product of the two tensors
    """
    data = left_tensor.data * right_tensor.data
    requires_gradient = left_tensor.requires_gradient or right_tensor.requires_gradient
    depends_on: List[Adjoint] = []

    if left_tensor.requires_gradient:
        def gradient_func_mul_left(gradient: 'np.ndarray') -> 'np.ndarray':
            gradient = gradient * right_tensor.data
            ndims_added = gradient.ndim - left_tensor.data.ndim

            for _ in range(ndims_added):
                gradient = gradient.sum(axis=0)

            for idx, dim in enumerate(left_tensor.shape):
                if dim == 1:
                    gradient = gradient.sum(axis=idx, keepdims=True)

            return gradient

        depends_on.append(Adjoint(left_tensor, gradient_func_mul_left))

    if right_tensor.requires_gradient:
        def gradient_func_mul_right(gradient: 'np.ndarray') -> 'np.ndarray':
            gradient = gradient * left_tensor.data

            ndims_added = gradient.ndim - right_tensor.data.ndim
            for _ in range(ndims_added):
                gradient = gradient.sum(axis=0)

            for idx, dim in enumerate(right_tensor.shape):
                if dim == 1:
                    gradient = gradient.sum(axis=idx, keepdims=True)

            return gradient

        depends_on.append(Adjoint(right_tensor, gradient_func_mul_right))

    return Tensor(data, requires_gradient, depends_on)

def _neg(tensor: 'Tensor') -> 'Tensor':
    """
    Returns the negated tensor
    """
    data = -tensor.data
    requires_gradient = tensor.requires_gradient
    depends_on: List[Adjoint] = []

    if requires_gradient:
        def gradient_func_neg(gradient: 'np.ndarray') -> 'np.ndarray':
            return -gradient

        depends_on.append(Adjoint(tensor, gradient_func_neg))

    return Tensor(data, requires_gradient, depends_on)

def _sub(left_tensor: 'Tensor', right_tensor: 'Tensor') -> 'Tensor':
    """
    Returns the difference between left_tensor and right_tensor repsectively
    """
    return left_tensor + -right_tensor

def _matmul(left_tensor: 'Tensor', right_tensor: 'Tensor') -> 'Tensor':
    """
    Returns the matrix product of the tensors
    """
    data = left_tensor.data @ right_tensor.data
    requires_gradient = left_tensor.requires_gradient or right_tensor.requires_gradient
    depends_on: List[Adjoint] = []

    if left_tensor.requires_gradient:
        def gradient_func_malmut_left(gradient: 'np.ndarray') -> 'np.ndarray':
            return gradient @ right_tensor.data.T

        depends_on.append(Adjoint(left_tensor, gradient_func_malmut_left))

    if right_tensor.requires_gradient:
        def gradient_func_malmut_right(gradient: 'np.ndarray') -> 'np.ndarray':
            return left_tensor.data.T @ gradient

        depends_on.append(Adjoint(right_tensor, gradient_func_malmut_right))

    return Tensor(data, requires_gradient, depends_on)

def _slice(tensor: 'Tensor', idxs: Union[List[int], int]) -> 'Tensor':
    data = tensor.data[idxs]
    requires_gradient = tensor.requires_gradient
    depends_on: List[Adjoint] = []

    if requires_gradient:
        def gradient_func_slice(gradient: 'np.ndarray') -> 'np.ndarray':
            padded_gradient = np.zeros(tensor.shape)
            padded_gradient[idxs] = gradient
            return padded_gradient

        depends_on.append(Adjoint(tensor, gradient_func_slice))

    return Tensor(data, requires_gradient, depends_on)

def _div(left_tensor: Tensor, right_tensor: Tensor) -> Tensor:
    data = left_tensor.data/right_tensor.data
    requires_gradient = left_tensor.requires_gradient or right_tensor.requires_gradient
    depends_on: List[Adjoint] = []

    if left_tensor.requires_gradient:
        def gradient_func_div_left(gradient: np.ndarray) -> np.ndarray:  # d(x/y)/dx = 1/y
            return gradient * (1/right_tensor.data)

        depends_on.append(Adjoint(left_tensor, gradient_func_div_left))

    if right_tensor.requires_gradient:
        def gradient_func_div_right(gradient: np.ndarray) -> np.ndarray:  # d(x/y)/dy = -x/y**2
            return gradient * -(left_tensor.data/(right_tensor.data)**2)

        depends_on.append(Adjoint(right_tensor, gradient_func_div_right))

    return Tensor(data, requires_gradient, depends_on)


def log(tensor: Tensor) -> Tensor:
    data = np.log(tensor.data)
    requires_gradient = tensor.requires_gradient
    depends_on: List[Adjoint] = []

    if requires_gradient:
        def gradient_func_neg(gradient: np.ndarray) -> np.ndarray:
            return gradient * 1/tensor.data

        depends_on.append(Adjoint(tensor, gradient_func_neg))

    return Tensor(data, requires_gradient, depends_on)

# def _pow(left_tensor: 'Tensor', right_tensor: 'Tensor') -> 'Tensor':