# Autodiff
Autodiff is a python framework for reverse mode automatic differentiation.

## Requirement installation

Installing requirements using the package manager [pip](https://pip.pypa.io/en/stable/).

```bash
pip install -r requirements.txt
```

## Usage

```python
from autodiff.tensor import Tensor

# simple tensor with gradient, prior to backward gradients are zero
t1 = Tensor(2., requires_gradient=True)
t1.gradient  # returns Tensor=0., requires_gradient=False

t2 = Tensor(10., requires_gradient=True)
t2.gradient  # returns Tensor=0., requires_gradient=False


# tensor product with backward pass to update gradients
t_prod = t1*t2
t_prod  # returns Tensor=20., requires_gradient=True
t_prod.backward()

t1.gradient  # returns Tensor=10.0, requires_gradient=False
t2.gradient  # returns Tensor=2.0, requires_gradient=False

```
## Examples
**xor (simple non-linear function):**

**1d gan:**


more to come..




## Milestones
**Tensor operations:**
- [x] sum
- [x] add
- [x] multiplication
- [x] negation
- [x] subtraction
- [x] transpose
- [x] slicing
- [x] division
- [x] power
- [x] natural logarithm
- [x] exp

**Network:**
- [x] Parameter
- [x] Module
- [x] Sequential
- [x] Linear Layer

**Activation functions:**
- [x] ReLU
- [x] Tanh

**Loss functions:**
- [ ] Binary Cross Entropy
- [x] Cross Entropy

more to come..
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Acknowledgment
I was inspired to try to write my own deep learning framework from (numpy) scratch. In part from working with the awesome Python library [PyTorch](https://pytorch.org/) and wanting to understand that better. As well as from the being inspired by [Joel Grus](https://joelgrus.com/), his talks
and YouTube videos on the matter.

Additionally, the book `Mathematics for Machine Learning` proved to be a very
helpful tool and an enjoyable read.

## License
[MIT](https://choosealicense.com/licenses/mit/)