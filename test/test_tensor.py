import unittest
import numpy as np

from typing import Tuple
from numpy.core.numeric import require

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
    def test_tensor_empty(self):
        Tensor([])

    def test_tensor_instantiation(self):
        Tensor(1., requires_gradient=True)
        Tensor([1, 2, 3], requires_gradient=True)
        Tensor(np.array([1, 2, 3]), requires_gradient=True)

    def test_tensor_backward_with_gradient(self):
        t = Tensor([1, 2, 3], requires_gradient=True)
        t.backward(Tensor(2.))
        tensor_assert(t.gradient, [2, 2, 2])

    def test_tensor_sum(self):
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
                     [4, 5, 6]], requires_gradient=True)
        t2 = Tensor([1, 1, 1], requires_gradient=True)

        (t1 + t2).backward(Tensor([1, 1, 1]))

        tensor_assert(t1.gradient, [[1, 1, 1],
                                    [1, 1, 1]])
        tensor_assert(t2.gradient, [1, 1, 1])

    def test_tensor_mul(self):
        t1 = Tensor([2, 4, 10], requires_gradient=True)
        t2 = Tensor([0.5, 0.25, 0.1], requires_gradient=True)
        t3 = t1 * t2
        t3.backward(Tensor(1.))

        tensor_assert(t1.gradient, t2.data)
        tensor_assert(t2.gradient, t1.data)
        tensor_assert(t3.gradient, [1, 1, 1])

    def test_tensor_mul_broadcasted(self):
        t1 = Tensor([[0.25, 0.1, 0.05],
                     [0.25, 0.1, 0.05]], requires_gradient=True)
        t2 = Tensor([2, 5, 10], requires_gradient=True)
        t3 = t1 * t2
        t3.backward(Tensor([1, 1, 1]))

        tensor_assert(t1.gradient, [t2.data, t2.data])
        tensor_assert(t2.gradient, [0.5, 0.2, 0.1])
        tensor_assert(t3.gradient, [[1, 1, 1],
                                    [1, 1, 1]])

    def test_tensor_neg(self):
        t1 = Tensor([1, 2, 3], requires_gradient=True)
        t2 = Tensor([1, 1, 1], requires_gradient=True)
        t3 = t1 * -t2
        t3.backward(Tensor(1.))

        tensor_assert(t1.gradient, [-1, -1, -1])
        tensor_assert(t2.gradient, [-1, -2, -3])

        tensor_assert(t3.data, [-1, -2, -3])
        tensor_assert(t3.gradient, [1, 1, 1])

    def test_tensor_sub(self):
        t1 = Tensor([1, 2, 3], requires_gradient=True)
        t2 = Tensor([2, 3, 4], requires_gradient=True)

        (t1 - t2).backward(Tensor([1.]))

        tensor_assert(t1.gradient, [1, 1, 1])
        tensor_assert(t2.gradient, [-1, -1, -1])

    def test_tensor_add_broadcasted(self):
        t1 = Tensor([[1, 2, 3],
                     [4, 5, 6]], requires_gradient=True)
        t2 = Tensor([1, 1, 1], requires_gradient=True)

        (t1 - t2).backward(Tensor(1.))

        tensor_assert(t1.gradient, [[1, 1, 1],
                                    [1, 1, 1]])
        tensor_assert(t2.gradient, [-1, -1, -1])

    def test_tensor_transpose(self):
        t = Tensor([[2],
                    [2]], requires_gradient=True)

        tensor_assert(t.T, [[2, 2]])
        assert t.depends_on == t.T.depends_on, 'Tensor transpose is not handling dependencies correctly'

    def test_tensor_matmul(self):
        t1 = Tensor([[1, 2, 3],
                     [4, 5, 6]], requires_gradient=True)  # (2, 3)

        t2 = Tensor([[10],
                     [10],
                     [10]], requires_gradient=True)  # (3, 1)

        t3 = t1 @ t2  # (2, 1)

        gradient = Tensor([[1],
                           [1]])
        t3.backward(gradient)

        tensor_assert(t3.data, [[60],
                               [150]])
        tensor_assert(t1.gradient, gradient @ t2.T)
        tensor_assert(t2.gradient, t1.T @ gradient)

    def test_tensor_slice(self):
        t = rtensor(100, 100, requires_gradient=True)
        assert t[50:, 50:].shape == (50, 50), 'Tensor slice is not handling slicing correctly'

    def test_tensor_slice_list(self):
        t = rtensor(100, 100, requires_gradient=True)
        assert t[[1, 2]].shape == (2, 100), 'Tensor slice is not handling list slicing correctly'

    def test_tensor_division(self):
        t1 = Tensor(2., requires_gradient=True)
        t2 = Tensor(10., requires_gradient=True)
        t3 = t1/t2
        t3.backward()

        tensor_assert(t3, [0.2])
        tensor_assert(t1.gradient, [0.1])
        tensor_assert(t2.gradient, [-0.02])