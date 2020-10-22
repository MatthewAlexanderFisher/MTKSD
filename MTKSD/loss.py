import torch
import torch.nn as nn
import numpy as np

import ot  # POT - Python Optimal Transport

l2 = nn.PairwiseDistance(2)  # this function l2 computes all the pairwise distances between each component of a tensor


# KSD U-statistic using inverse multiquadric kernel

def KSD_U(samples, score_func, gamma=1):
    N, d = samples.size()  # number of samples, dimension
    g = (1 / gamma) ** 2

    scores = score_func(samples)
    s1 = scores.repeat(1, N).view(N * N, d)
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


# KSD U-statistic using inverse multiquadric kernel

def KSD_U_nograd(samples, score_func, gamma=1):
    N, d = samples.size()  # number of samples, dimension
    g = (1 / gamma) ** 2

    scores = score_func(samples)
    s1 = scores.repeat(1, N).view(N * N, d)
    s2 = scores.repeat(N, 1)

    diffs = (samples.unsqueeze(1) - samples).reshape(N * N, d)
    dists = torch.cdist(samples, samples).flatten() ** 2

    k = (1 + g * dists) ** (-1 / 2)
    k_x = -g * (1 + g * dists[:, None]) ** (-3 / 2) * diffs
    k_xy = -3 * g ** 2 * dists * (1 + g * dists) ** (-5 / 2) + g * d * (1 + g * dists) ** (-3 / 2)

    outvec = k * torch.sum(s1 * s2, dim=-1) + torch.sum(-s1 * k_x, dim=-1) + torch.sum(s2 * k_x, dim=-1) + k_xy
    diag = torch.sum(scores ** 2, dim=-1) + g * d

    with torch.no_grad():
        output = 1 / (N * (N - 1)) * (torch.sum(outvec) - torch.sum(diag))

    return output


# KSD U-statistic using inverse multiquadric kernel

def KSD_gammaU(samples, score_func, gamma):
    N, d = samples.size()  # number of samples, dimension
    g = 1 / gamma

    scores = score_func(samples)
    s1 = scores.repeat(1, N).view(N * N, d)
    s2 = scores.repeat(N, 1)

    g_samps = g * samples
    diffs = ((g * g_samps).unsqueeze(1) - g * g_samps).reshape(N * N, d)
    dists = torch.cdist(g_samps, g_samps).flatten() ** 2
    dists2 = torch.cdist(g * g_samps, g * g_samps).flatten() ** 2

    k = (1 + dists) ** (-1 / 2)
    k_x = -(1 + dists[:, None]) ** (-3 / 2) * diffs
    k_xy = -3 * dists2 * (1 + dists) ** (-5 / 2) + torch.sum(g ** 2) * (1 + dists) ** (-3 / 2)

    outvec = k * torch.sum(s1 * s2, dim=-1) + torch.sum(-s1 * k_x, dim=-1) + torch.sum(s2 * k_x, dim=-1) + k_xy
    diag = torch.sum(scores ** 2, dim=-1) + torch.sum(g ** 2)

    output = 1 / (N * (N - 1)) * (torch.sum(outvec) - torch.sum(diag))

    return output


# KSD V-statistic using inverse multiquadric kernel

def KSD_V(samples, score_func, gamma=1):
    N, d = samples.size()  # number of samples, dimension
    g = (1 / gamma) ** 2

    scores = score_func(samples)
    s1 = scores.repeat(1, N).view(N * N, d)
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
    d_matrix = ot.dist(samp1.detach().numpy(), samp2.detach().numpy(), metric="euclidean")  # W_1

    n = len(samp1.detach().numpy())

    w1 = np.zeros(n) + 1 / n
    w2 = np.zeros(n) + 1 / n

    dist = ot.emd2(w1, w2, d_matrix, numItermax=1000000)  # Earth movers distance (this is W_2 ** 2)

    return dist


# ELBO for polynomial transport

def ELBO_polynomial(x, theta, T_x, target_log_prob, order):
    x_1 = x[:, 0]
    x_2 = x[:, 1]

    theta_1 = theta[0:(order + 1)]
    theta_2 = theta[(order + 1):]

    grad_T_x_1 = 0.
    l_range = torch.tensor(range(order)) + 1
    for l in l_range:
        grad_T_x_1 += l * theta_1[l] * (x_1 ** l)

    grad_T_x_2 = 0.
    count_start = 0
    l_range = torch.tensor(range(order))
    for l in l_range:

        count_end = count_start + order + 1 - l
        theta_2_l = theta_2[count_start:count_end]
        k_range = torch.tensor(range(order - l)) + 1
        for k in k_range:
            grad_T_x_2 += k * theta_2_l[k] * (x_1 ** l) * (x_2 ** (k - 1))

        count_start = count_end

    log_abs_det_grad_T_x = grad_T_x_1.abs().log() + grad_T_x_2.abs().log()
    KLD_val = -(target_log_prob(T_x) + log_abs_det_grad_T_x).mean()

    return KLD_val
