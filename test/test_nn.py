import unittest
import numpy as np

from typing import Tuple
from numpy.matrixlib import defmatrix

from numpy.testing import assert_array_almost_equal
from autodiff.nn import Parameter, ReLU, Tanh
from autodiff.tensor import Tensor, Tensorable

def tensor_assert(
    t1: Tensorable,
    t2: Tensorable,
    decimal: int = 4,
    err_msg: str = '',
    verbose: bool = True
) -> None:
    """Raises assert error if two tensors are not equal up to specified
    decimal precision"""
    if type(t1) == Tensor:
        t1 = t1.data
    if type(t2) == Tensor:
        t2 = t2.data

    assert_array_almost_equal(t1, t2, decimal, err_msg, verbose)

class TestParameter(unittest.TestCase):
    def test_parameter_empty(self):
        p = Parameter()
        assert p.shape == ()
    
    def test_parameter_multi_shape(self):
        shape = (10, 10, 10)
        p = Parameter(*shape)
        assert p.shape == shape

    def test_parameter_normal(self):
        shape = (10, 10, 10)
        p = Parameter(*shape, dist='normal')
        assert p.shape == shape

class TestLayer(unittest.TestCase):
    pass

class TestActivations(unittest.TestCase):
    def test_tanh(self):
        grad = Tensor([1., 1., 1.])
        g = Tanh()

        t1 = Tensor([-1., -1., -1.], requires_gradient=True)
        t2 = Tensor([0., 0., 0.], requires_gradient=True)
        t3 = Tensor([2., 2, 2], requires_gradient=True)
        a1 = g(t1)
        a2 = g(t2)
        a3 = g(t3)
        o = a1 + a2 + a3
        o.backward(grad)

        tensor_assert(a1, [-0.7616, -0.7616, -0.7616])
        tensor_assert(a2, [0., 0., 0.])
        tensor_assert(a3, [0.9640, 0.9640, 0.9640])

        tensor_assert(t1.gradient, [0.4200, 0.4200, 0.4200])
        tensor_assert(t2.gradient, [1., 1., 1.])
        tensor_assert(t3.gradient, [0.0707, 0.0707, 0.0707])

    def test_relu(self):
        grad = Tensor([1., 1., 1.])
        g = ReLU()

        t1 = Tensor([-1., -1., -1.], requires_gradient=True)
        t2 = Tensor([0., 0., 0.], requires_gradient=True)
        t3 = Tensor([2., 2, 2], requires_gradient=True)
        a1 = g(t1)
        a2 = g(t2)
        a3 = g(t3)
        o = a1 + a2 + a3
        o.backward(grad)

        tensor_assert(a1, [0, 0, 0])
        tensor_assert(a2, [0, 0, 0])
        tensor_assert(a3, [2, 2, 2])

        tensor_assert(t1.gradient, [0, 0, 0])
        tensor_assert(t2.gradient, [0, 0, 0])
        tensor_assert(t3.gradient, [1, 1, 1])
