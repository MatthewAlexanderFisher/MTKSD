import torch
import torch.nn as nn
import numpy as np

import ot  # POT - Python Optimal Transport

l2 = nn.PairwiseDistance(2)  # this function l2 computes all the pairwise distances between each component of a tensor


# KSD U-statistic using inverse multiquadric kernel

def KSD_U(samples, score_func, gamma=1):
    N, d = samples.size()  # number of samples, dimension
    g = 1 / gamma

    scores = score_func(samples)
    s1 = scores.repeat(1, N).view(N * N, 2)
    s2 = scores.repeat(N, 1)

    diffs = (samples.unsqueeze(1) - samples).reshape(N * N, d)
    dists = torch.cdist(samples, samples).flatten() ** 2

    k = (1 + g * dists) ** (-1 / 2)
    k_x = -g * (1 + g * dists[:, None]) ** (-3 / 2) * diffs
    k_xy = -3 * g ** 2 * dists * (1 + g * dists) ** (-5 / 2) + g * d * (1 + g * dists) ** (-3 / 2)

    outvec = k * torch.sum(s1 * s2, dim=-1) + torch.sum(-s1 * k_x, dim=-1) + torch.sum(s2 * k_x, dim=-1) + k_xy
    diag = torch.sum(scores ** 2, dim=-1) + g * d

    output = 1 / (N * (N - 1)) * (torch.sum(outvec) - torch.sum(diag))

    return output


# KSD V-statistic using inverse multiquadric kernel

def KSD_V(samples, score_func, gamma=1):
    N, d = samples.size()  # number of samples, dimension
    g = 1 / gamma

    scores = score_func(samples)
    s1 = scores.repeat(1, N).view(N * N, 2)
    s2 = scores.repeat(N, 1)

    diffs = (samples.unsqueeze(1) - samples).reshape(N * N, d)
    dists = torch.cdist(samples, samples).flatten() ** 2

    k = (1 + g * dists) ** (-1 / 2)
    k_x = -g * (1 + g * dists[:, None]) ** (-3 / 2) * diffs
    k_xy = -3 * g ** 2 * dists * (1 + g * dists) ** (-5 / 2) + g * d * (1 + g * dists) ** (-3 / 2)

    outvec = k * torch.sum(s1 * s2, dim=-1) + torch.sum(-s1 * k_x, dim=-1) + torch.sum(s2 * k_x, dim=-1) + k_xy
    output = torch.mean(outvec)

    return output


# KSD 2-sample statistic using inverse multiquadric kernel

def KSD_2sample(sample1, sample2, score_func, gamma=1):
    d = sample1.size()[1]
    g = 1 / gamma

    s1 = score_func(sample1)
    s2 = score_func(sample2)

    diffs = sample1 - sample2
    dists = l2(sample1, sample2) ** 2

    k = (1 + g * dists) ** (-1 / 2)
    k_x = -g * (1 + g * dists[:, None]) ** (-3 / 2) * diffs
    k_y = -k_x
    k_xy = -3 * g ** 2 * dists * (1 + g * dists) ** (-5 / 2) + g * d * (1 + g * dists) ** (-3 / 2)

    outvec = k * torch.sum(s1 * s2, dim=-1) + torch.sum(s1 * k_y, dim=-1) + torch.sum(s2 * k_x, dim=-1) + k_xy
    output = torch.mean(outvec)

    return output

# ELBO / KL divergence (ELBO assumes given distributions have unnormalised log densities)

def ELBO(approx_dist, target_dist, sample):
    return torch.mean(approx_dist.log_prob(sample) - target_dist.log_prob(sample))


# Wasserstein distance (W_2 ** 2) To change to W_1 use ot.dist

def Wasserstein(samp1, samp2):
    samp1_np = samp1.detach().numpy()
    samp2_np = samp2.detach().numpy()
    d_matrix = ot.dist(samp1_np, samp2_np)

    # d_matrix = ot.dist(samp1.numpy(), samp2.numpy(), metric="euclidean")

    n = len(samp1_np)

    w1 = np.zeros(n) + 1 / n
    w2 = np.zeros(n) + 1 / n

    dist = ot.emd2(w1, w2, d_matrix)  # Earth movers distance (I think this is W_2)

    return dist
