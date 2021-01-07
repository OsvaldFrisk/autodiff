import unittest
import numpy as np

from numpy.testing import assert_array_almost_equal as tensor_assert
from autodiff.tensor import Tensor

class TestTensor(unittest.TestCase):
    def test_empty_tensor(self):
        Tensor([])
       
    def test_tensor_instantiation(self):
        Tensor(1., requires_gradient=True)
        Tensor([1, 2, 3], requires_gradient=True)
        Tensor(np.array([1, 2, 3]), requires_gradient=True)