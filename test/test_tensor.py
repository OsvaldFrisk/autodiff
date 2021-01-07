import unittest
import numpy as np

from typing import Tuple

from numpy.testing import assert_array_almost_equal
from autodiff.tensor import Tensor, Tensorable


def rtensor(*shape: Tuple, requires_gradient: bool = True) -> Tensor:
    """Instansites a tensor with randomly sampled values from uniform
    distribution between [0,1] with specified shape"""
    return Tensor(np.random.rand(*shape), requires_gradient=requires_gradient)


def tensor_assert(
    t1: Tensorable,
    t2: Tensorable,
    decimal: int = 6,
    err_msg: str = '',
    verbose: bool = True
) -> None:
    """Raises assert error if two tensors are not equal up to specified
    decimal precision"""
    if type(t1) == Tensor:
        t1 = t1.data
    if type(t2) == Tensor:
        t2 = t2.data

    assert_array_almost_equal(t1, t2)


class TestTensor(unittest.TestCase):
    def test_empty_tensor(self):
        Tensor([])

    def test_tensor_instantiation(self):
        Tensor(1., requires_gradient=True)
        Tensor([1, 2, 3], requires_gradient=True)
        Tensor(np.array([1, 2, 3]), requires_gradient=True)

    def test_tensor_backward_with_gradient(self):
        t = Tensor([1, 2, 3], requires_gradient=True)
        t.backward(Tensor(2.))
        tensor_assert(t.gradient, [2, 2, 2])

    def test_tensoroperator_sum(self):
        t = Tensor([1, 2, 3], requires_gradient=True)
        t.sum().backward()
        tensor_assert(t.gradient, [1, 1, 1])

    def test_tensor_add(self):
        t1 = Tensor([1, 2, 3], requires_gradient=True)
        t2 = Tensor([2, 3, 4], requires_gradient=True)

        (t1 + t2).backward(Tensor([1., 2., 3.]))

        tensor_assert(t1.gradient, [1, 2, 3])
        tensor_assert(t2.gradient, [1, 2, 3])

    def test_tensor_add_broadcasted(self):
        t1 = Tensor([[1, 2, 3],
                     [4, 5, 6]],
                    requires_gradient=True)
        t2 = Tensor([1, 1, 1], requires_gradient=True)

        (t1 + t2).backward(Tensor([1, 1, 1]))

        tensor_assert(t1.gradient, [[1, 1, 1],
                                    [1, 1, 1]])
        tensor_assert(t2.gradient, [1, 1, 1])
