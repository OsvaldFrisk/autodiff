import numpy as np
import inspect
from typing import Callable, Dict, List, Sequence, Tuple, Union



from autodiff.tensor import Tensor, Adjoint


class Parameter(Tensor):
    def __init__(self, *shape: Tuple[int, ...], dist: str = 'uniform') -> Tensor:
        data: np.ndarray

        if dist == 'uniform':
            data = np.random.rand(*shape)
        elif dist == 'normal':
            data = np.random.randn(*shape)
        else:
            raise ValueError('distribution type is not recognized')

        super().__init__(data, requires_gradient=True)

class Layer:
    def __init__(self):
        self.params: Dict[str, Parameter] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError


class Activation:
    def __init__(self, F: Callable[[Tensor], Tensor]) -> None:
        self.F = F

    def forward(self, inputs: Tensor) -> Tensor:
        return self.F(inputs)

    def __call__(self, inputs: Tensor) -> Tensor:
        return self.forward(inputs)

class Linear(Layer):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.params['w'] = Parameter(input_size, output_size)
        self.params['b'] = Parameter(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        z = inputs @ self.params['w'] + self.params['b']
        return z

class Module:
    def __init__(self):
        self.list_of_params: List[Parameter] = []
    def parameters(self) -> List[Parameter]:
        for (key, value) in inspect.getmembers(self):
            if isinstance(value, Layer):
                self.list_of_params.append(value.params)

            elif isinstance(value, Module):
                self.list_of_params.append(value.parameters())
        return self.list_of_params

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_gradient()

class Sequential(Module):
    def __init__(self, layers: Sequence[Union[Layer, Callable]]) -> None:
        super().__init__()
        self.layers = layers

    def parameters(self) -> List[Parameter]:
        for layer in self.layers:
            if isinstance(layer, Layer):
                self.list_of_params.append(layer.params)

            elif isinstance(layer, Sequential):
                sub_list_of_params = layer.parameters()
                for param in sub_list_of_params:
                    self.list_of_params.append(param)
        return self.list_of_params

    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def __call__(self, inputs: Tensor) -> Tensor:
        return self.forward(inputs)

class Tanh(Activation):
    def __init__(self):
        super().__init__(_tanh)

def _tanh(tensor: Tensor) -> Tensor:
    data = np.tanh(tensor.data)
    requires_gradient = tensor.requires_gradient
    depends_on: List[Adjoint] = []

    if requires_gradient:
        def gradient_func_tanh(gradient: Tensor) -> Tensor:
            return gradient * (1 - data * data)

        depends_on.append(Adjoint(tensor, gradient_func_tanh))

    return Tensor(data, requires_gradient, depends_on)

class ReLU(Activation):
    def __init__(self):
        super().__init__(_relu)


def _relu(inputs: Tensor) -> Tensor:
    relu_mask = inputs.data > 0
    data = inputs.data * relu_mask
    requires_gradient = inputs.requires_gradient
    depends_on: List[Adjoint] = []

    if requires_gradient:
        def gradient_func_relu(gradient: Tensor) -> Tensor:
            return gradient * relu_mask
        depends_on.append(Adjoint(inputs, gradient_func_relu))

    return Tensor(data, requires_gradient, depends_on)

class CrossEntropyLoss:
    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        loss = cross_entropy(predictions, targets)
        requires_gradient = predictions.requires_gradient
        depends_on: List[Adjoint] = []

        if requires_gradient:
            def gradient_func_cross_entropy(gradient: Tensor) -> Tensor:
                """
                https://deepnotes.io/softmax-crossentropy#derivative-of-cross-entropy-loss-with-softmax
                X is the output from fully connected layer (num_examples x num_classes)
                y is labels (num_examples x 1)
                    Note that y is not one-hot encoded vector.
                    It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
                """
                N = targets.shape[0]
                dL = stable_softmax(predictions.data)
                dL[range(N), targets.data] -= 1
                dL = dL/N
                return gradient * dL

            depends_on.append(Adjoint(predictions, gradient_func_cross_entropy))

        return Tensor(loss, True, depends_on)


def stable_softmax(X):
    """
    A more stable version of softmax,
    e.g. X = [1000, 2000, 3000] results in [0, 0, 1] and not [nan, nan, nan]
    """
    exps = np.exp(X.data - np.max(X.data, axis=1).reshape(-1, 1))
    # exps = np.exp(X.data)
    return exps / np.sum(exps)

def cross_entropy(X,y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    https://deepnotes.io/softmax-crossentropy#derivative-of-cross-entropy-loss-with-softmax
    """
    assert len(y.shape) < 3, "y has incorrect dimensions"
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    p = stable_softmax(X) + 1e-9

    N = y.shape[0]
    log_likelihood = -np.log(p[range(N),y.data])
    loss = np.sum(log_likelihood) / N
    return loss
