import unittest
import numpy as np

from numpy.testing import assert_array_almost_equal
from autodiff.tensor import Tensor, Tensorable


def tensor_assert(
    t1: Tensorable,
    t2: Tensorable,
    decimal: int = 6,
    err_msg: str = '',
    verbose: bool = True
) -> None:
    """Raises assert error if two tensors are not equal up to specified decimal precision"""
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
        tensor_assert(t.gradient.data, [2, 2, 2])

    def test_tensoroperator_sum(self):
        t = Tensor([1, 2, 3], requires_gradient=True)
        t.sum().backward()
        tensor_assert(t.gradient, [1, 1, 1])

    def test_tensor_add(self):
        t1 = Tensor([1, 2, 3], requires_gradient=True)
        t2 = Tensor([2, 3, 4], requires_gradient=True)

        (t1 + t2).backward(Tensor([1., 2., 3.]))

        tensor_assert(t1.gradient.data, [1, 2, 3])
        tensor_assert(t2.gradient.data, [1, 2, 3])

    def test_tensor_add_broadcasted(self):
        t1 = Tensor([[1, 2, 3],
                     [4, 5, 6]],
                     requires_gradient=True)
        t2 = Tensor([1, 1, 1], requires_gradient=True)

        (t1 + t2).backward(Tensor([1, 1, 1]))

        tensor_assert(t1.gradient.data, [[1, 1, 1],
                                         [1, 1, 1]])
        tensor_assert(t2.gradient.data, [1, 1, 1])