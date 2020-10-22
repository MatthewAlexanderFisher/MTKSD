import torch

import matplotlib.pyplot as plt
import numpy as np


# plots the loss from stochastic gradient descent along with a smoothed running average

def plot_loss(loss_vec, n=10, log=True):
    if log is True:
        loss_vec_np = torch.log(torch.abs(loss_vec.detach())).numpy()
    else:
        loss_vec_np = loss_vec.detach().numpy()

    count_array = np.arange(loss_vec_np.size) + 1

    def moving_average(a, N):
        ret = np.cumsum(a, dtype=float)
        ret[N:] = ret[N:] - ret[:-N]
        return ret[N - 1:] / N

    mov_average = moving_average(loss_vec_np, n)

    plt.plot(count_array, loss_vec_np)
    plt.plot(count_array[:-n + 1], mov_average)


# plots the contour plots of a 2D distribution object

def plot_dist2D(dist, x=[-10, 10], y=[-10, 10], n_levels=200, cmap="magma", n_steps=300):
    xline = torch.linspace(x[0], x[1], steps=n_steps)
    yline = torch.linspace(y[0], y[1], steps=n_steps)
    xgrid, ygrid = torch.meshgrid(xline, yline)
    xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

    with torch.no_grad():
        zgrid = dist.log_prob(xyinput).exp().reshape(n_steps, n_steps)

    plt.contourf(xgrid.numpy(), ygrid.numpy(), zgrid.numpy(), levels=n_levels, cmap=cmap)


# returns components of dist 2D

def get_distvals(dist, x=[-10, 10], y=[-10, 10], n_steps=300):

    xline = torch.linspace(x[0], x[1], steps=n_steps)
    yline = torch.linspace(y[0], y[1], steps=n_steps)
    xgrid, ygrid = torch.meshgrid(xline, yline)
    xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

    with torch.no_grad():
        zgrid = dist.log_prob(xyinput).exp().reshape(n_steps, n_steps)

    return xgrid.numpy(), ygrid.numpy(), zgrid.numpy()


# plots the scatter plot of a pytorch sample

def plot_scatter(sample, color="firebrick", alpha=0.1):
    sampleT = sample.T.numpy()
    plt.scatter(sampleT[0], sampleT[1], color=color, alpha=alpha)


# plots the solutions of predator prey model

def plot_solutions(sols, t):
    sols = torch.squeeze(sols).T
    t = t.numpy()
    for i in range(sols.shape[0]):
        sol_i = sols[i].numpy()
        plt.plot(t, sol_i, label="y" + str(i + 1))
    plt.legend(loc="upper right")
