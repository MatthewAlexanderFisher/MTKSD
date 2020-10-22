import torch
import torch.nn as nn
import torch.distributions as D
import numpy as np

from .get_score import get_score


# Mixture of Gaussians

class MOG2D:

    def __init__(self, mean, var):
        self.mean = mean
        self.var = var
        self.dist = self.get_dist()

    def sample(self, n):
        return self.dist.sample((n,))

    def log_prob(self, x):
        return self.dist.log_prob(x)

    def score(self, x):
        return get_score(x, self.dist)

    def get_dist(self):
        n = len(self.mean)
        mix = D.Categorical(torch.ones(n, ))
        comp = D.Independent(D.Normal(self.mean, self.var * torch.ones(n, 2)), 1)

        return D.MixtureSameFamily(mix, comp)


# Banana distribution

class Banana2D:

    def __init__(self, var):
        self.var = var
        self.dist = D.MultivariateNormal(torch.Tensor([0, 0]), torch.eye(2))

    def sample(self, n):
        a, v1, v2 = self.var
        samp = self.dist.sample(torch.Size([n, ]))
        sampt = samp.t()
        sampt[0] = sampt[0] * v1
        sampt[1] = sampt[1] * v2 + a * sampt[0].pow(2)
        return sampt.t()

    def log_prob(self, x):
        a, v1, v2 = self.var
        n = torch.Tensor([np.pi * 2 * v1 * v2])
        x1, x2 = x.T
        output = -torch.log(n) - 0.5 * (x1 ** 2 / v1 ** 2 + (x2 - a * x1 ** 2) ** 2 / v2 ** 2)
        return output

    def score(self, x):
        a, v1, v2 = self.var
        x1, x2 = x.t()
        dx1 = -x1 / v1 ** 2 + 2 * a * x1 * (x2 - a * x1 ** 2) / v2 ** 2
        dx2 = -(x2 - a * x1 ** 2) / v2 ** 2
        return torch.stack((dx1, dx2)).t()


# Sinusoidal distribution

class Sinusoidal2D:

    def __init__(self, var):
        self.var = var
        self.dist = D.MultivariateNormal(torch.Tensor([0, 0]), torch.eye(2))

    def sample(self, n):
        a, v1, v2 = self.var
        samp = self.dist.sample(torch.Size([n, ]))
        sampt = samp.t()
        sampt[0] = sampt[0] * v1
        sampt[1] = sampt[1] * v2 + torch.sin(a * sampt[0])
        return sampt.t()

    def log_prob(self, x):
        a, v1, v2 = self.var
        n = torch.Tensor([2 * np.pi * v1 * v2])
        x1, x2 = x.T
        output = -torch.log(n) - 0.5 * (x1 ** 2 / v1 ** 2 + (x2 - torch.sin(a * x1)) ** 2 / v2 ** 2)
        return output

    def score(self, x):
        a, v1, v2 = self.var
        x1, x2 = x.t()
        dx1 = - x1 / v1 ** 2 + a * torch.cos(a * x1) * (x2 - torch.sin(a * x1)) / v2 ** 2
        dx2 = - (x2 - torch.sin(a * x1)) / v2 ** 2
        return torch.stack((dx1, dx2)).t()