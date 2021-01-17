"""1D Generative Adversarial Network example, inspired by Jason Brownlee's awesome 1D gan
(https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-1-dimensional-function-from-scratch-in-keras/)
So I decided to implement it in autodiff.

Disclaimer: To evaluate the generator network you will need the matplotlib
plotting library, in addition to a directory called "gan_imgs".
"""

import matplotlib.pyplot as plt
from typing import Callable, Tuple
import numpy as np
from autodiff.tensor import Tensor
from autodiff import nn
from autodiff import optim




def train(
    X: Tensor,
    y: Tensor,
    net: nn.Module,
    optimizer: optim.Optimizer,
    loss_func: Callable,
    epochs: int,
    batch_size: int
    ) -> None:
    """trains network, minimizing loss using the optimizer"""

    for epochs in range(epochs):
        epoch_loss = 0.0

        for start in range(0, 100, batch_size):
            end = start + batch_size

            optimizer.zero_grad()

            inputs = X[start:end]
            preds = net.forward(inputs)
            targets = y[start:end].reshape(preds.shape)

            loss = loss_func(preds, targets)

            loss.backward()
            epoch_loss += loss.data

            optimizer.step()
            optimizer.zero_grad()

def eval_gen(
    Z: Tensor,
    gen: nn.Module,
    ) -> None:
    """prints networks prediction accuracy on test dataset"""
    x_plot = np.linspace(-0.5, 0.5, 10)
    y_plot = x_plot**2

    Z = gen.forward(z_plot)

    fig, ax = plt.subplots(1, 1)
    k = 1
    ax.set_xlim(-0.6*k, 0.6*k)
    ax.set_ylim(-0.2*k, 0.5*k)
    ax.plot(x_plot, y_plot, 'r', linewidth='3')
    ax.scatter(Z[:,0].data, Z[:,1].data)
    fig.savefig(f'./gan_imgs/{epoch+1}.png')

    plt.close()



def SE(preds: Tensor, targets: Tensor) -> Tensor:
    E = targets - preds
    return (E*E).sum()

def dis_real_data(n: int) -> Tuple[Tensor, Tensor]:
    x1 = np.random.rand(n, 1) - 0.5
    x2 = x1**2
    Xr = Tensor(np.hstack((x1, x2)))
    real_labels = np.zeros((n, 2))
    real_labels[:,1] = 1
    yr = Tensor(real_labels)
    return Xr, yr

def dis_fake_data(n: int, gen: nn.Module) -> Tuple[Tensor, Tensor]:
    Xf = gen.forward(Tensor(np.random.rand(n, 5) - 0.5))
    fake_labels = np.zeros((n, 2))
    fake_labels[:,0] = 1
    yf = Tensor(fake_labels)
    return Xf, yf

def gen_data(n: int) -> Tuple[Tensor, Tensor]:
    Xg = Tensor(np.random.rand(n, 5) - 0.5)
    fake_labels = np.zeros((n, 2))
    fake_labels[:,1] = 1
    yg = Tensor(fake_labels)
    return Xg, yg

dis = nn.Sequential([
    nn.Linear(2, 25),
    nn.Tanh(),
    nn.Linear(25, 2),
])

gen = nn.Sequential([
    nn.Linear(5, 50),
    nn.Tanh(),
    nn.Linear(50, 2),
    # nn.Tanh()

])

gan = nn.Sequential([
    gen,
    dis
])

# defining hyper-parameters
dis_optim = optim.SGD(dis.parameters, lr=0.0002)
gen_optim = optim.SGD(gen.parameters, lr=0.0001)
batch_size = 20
epochs = 25000
loss_func = SE

z_plot, _ = gen_data(50)

for epoch in range(epochs):
    N = 200
    Xr, yr = dis_real_data(N)
    Xf, yf = dis_fake_data(N, gen)
    Xg, yg = gen_data(N*2)

    train(Xr, yr, dis, dis_optim, SE, 1, batch_size)
    train(Xf, yf, dis, dis_optim, SE, 1, batch_size)
    train(Xg, yg, gan, gen_optim, SE, 1, batch_size)


    if (epoch + 1) % 50 == 0:
        eval_gen(z_plot, gen)

