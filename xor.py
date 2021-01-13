"""A simple example of a non-linear function approximation using autodiff
"""
from autodiff.tensor import Tensor
from autodiff.nn import Sequential, Linear, Tanh, ReLU
from autodiff.optim import SGD

# xor binary inputs
X = Tensor([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

# xor truth-table
y = Tensor([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0],
])

# simple sequential network with non-linear activation function
net = Sequential([
    Linear(2, 2),
    Tanh(),
    Linear(2, 2)
])

# training/optimizing using stochastic gradient decent
optimizer = SGD(net.parameters(), lr=0.05)
for epoch in range(5000):
    optimizer.zero_grad()
    pred = net(X)

    errors = pred - y.reshape(pred.shape)
    loss = (errors * errors).sum()

    loss.backward()
    optimizer.step()

    if (epoch+1) % 200 == 0:
        print(f"Epoch: {epoch+1}, Loss: {loss.data}")

print()
print("=" * 50)
print(f"Network prediction for xor truthtable:")
print(net(X).data)
