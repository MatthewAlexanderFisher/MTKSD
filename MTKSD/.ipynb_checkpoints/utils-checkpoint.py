import torch

import numpy as np
import matplotlib.pyplot as plt

from .loss import KSD_U, KSD_gammaU, ELBO, Wasserstein, KSD_U_nograd, KSD_V
from .plot import plot_loss

import pickle
import os
import time


def train_KSD(model, target, transform, transform_name, gamma=0.1, n_steps=10000, save_out=False, m=200,
              print_loss=False, lr=1e-3, V=False):
    loss_vec = torch.zeros(n_steps)

    timings = np.zeros(n_steps // m)
    start = time.time()
    transform_samples = []
    iter_num = []

    optimizer = torch.optim.Adam(transform.parameters(), lr=lr)  # change

    for i in range(n_steps):
        optimizer.zero_grad()

        base_dist_sample = model.base_dist.sample(torch.Size([100, ]))
        sample = model.transforms[0](base_dist_sample)  # change

        if V is False:
            loss = KSD_U(sample, target.score, gamma)
        else:
            loss = KSD_V(sample, target.score, gamma)

        loss.backward()

        optimizer.step()

        loss_vec[i] = loss.clone().detach()

        if (i + 1) % m == 0 and save_out is True:
            timings[i // m] = time.time() - start
            transform_samples.append(model.sample((10000,)))
            iter_num.append(i)

        model.clear_cache()  # change

        if (i + 1) % (n_steps // 2) == 0 and print_loss is True:
            print("iteration: " + str(i + 1) + ", loss: " + str(loss))
            plot_loss(loss_vec[:i], 200, log=False)
            plt.show()

    if save_out is True:
        save_output([timings, transform_samples, iter_num], "KSD_" + transform_name, "output")

    return model


def train_ELBO(model, target, transform, transform_name, n_steps=10000, save_out=False, m=200,
               print_loss=False, lr=1e-3):
    loss_vec = torch.zeros(n_steps)

    timings = np.zeros(n_steps // m)
    start = time.time()
    transform_samples = []
    iter_num = []

    optimizer = torch.optim.Adam(transform.parameters(), lr=lr)  # change

    for i in range(n_steps):
        optimizer.zero_grad()

        base_dist_sample = model.base_dist.sample(torch.Size([100, ]))
        sample = model.transforms[0](base_dist_sample)

        loss = ELBO(model, target, sample)
        loss.backward()

        optimizer.step()

        loss_vec[i] = loss.clone().detach()

        if (i + 1) % m == 0 and save_out is True:
            timings[i // m] = time.time() - start
            transform_samples.append(model.sample((10000,)))
            iter_num.append(i)

        model.clear_cache()

        if (i + 1) % (n_steps // 5) == 0 and print_loss is True:
            print("iteration: " + str(i + 1) + ", loss: " + str(loss))
            plot_loss(loss_vec[:i], 200, log=False)
            plt.show()

    if save_out is True:
        save_output([timings, transform_samples, iter_num], "ELBO_" + transform_name, "output")

    return model


def train_gammaKSD(model, target, transform, transform_name, gamma, n_steps=10000, save_out=False, m=200,
                   print_loss=False, lr=1e-3):
    loss_vec = torch.zeros(n_steps)

    timings = np.zeros(n_steps // m)
    start = time.time()
    transform_samples = []
    iter_num = []

    optimizer = torch.optim.Adam(transform.parameters(), lr=lr)  # change

    for i in range(n_steps):
        optimizer.zero_grad()

        base_dist_sample = model.base_dist.sample(torch.Size([100, ]))
        sample = model.transforms[0](base_dist_sample)  # change

        loss = KSD_gammaU(sample, target.score, gamma)
        loss.backward()

        optimizer.step()

        loss_vec[i] = loss.clone().detach()

        if (i + 1) % m == 0 and save_out is True:
            timings[i // m] = time.time() - start
            transform_samples.append(model.sample((10000,)))
            iter_num.append(i)

        model.clear_cache()  # change

        if (i + 1) % (n_steps // 5) == 0 and print_loss is True:
            print("iteration: " + str(i + 1) + ", loss: " + str(loss))
            plot_loss(loss_vec[:i], 200)
            plt.show()

    if save_out is True:
        save_output([timings, transform_samples, iter_num], "KSD_" + transform_name, "output")

    return model


def save_output(out, filename, dir):
    cwd = os.getcwd()
    directory = os.path.join(cwd, dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("Directory ", directory, "created")
    with open(dir + '/' + filename + ".pickle", 'wb') as handle:
        pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("saved output ", filename, " to ", directory)


def load_output(filename, dir):
    with open(dir + '/' + filename + ".pickle", 'rb') as handle:
        output = pickle.load(handle)
    return output


# performance metrics

def get_metric(model, target, target_samps, gamma=1, seed=0):
    torch.manual_seed(seed)
    final_samps = model.sample((10000,)).detach()
    wass = Wasserstein(target_samps, final_samps)

    final_samps.requires_grad_(True)
    ksd_u = KSD_U_nograd(final_samps, target.score, gamma=gamma)

    return wass, ksd_u
