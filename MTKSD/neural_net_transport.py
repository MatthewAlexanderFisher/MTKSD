import torch
import torch.nn as nn
import torch.nn.functional as F


class transform_dist:
    def __init__(self, base_dist, transforms):
        self.base_dist = base_dist
        self.transforms = transforms

    def sample(self, n):
        samp = self.base_dist.sample(n)
        for i in self.transforms:
            samp = i(samp)
        return samp

    def clear_cache(self):
        return ()


class ReLU_transport(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dims, hidden_n):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input = nn.Linear(input_dim, hidden_dims[0])
        self.linears1 = ([nn.Linear(hidden_dims[0], hidden_dims[0]) for i in range(hidden_n[0] - 1)])
        self.mid_in = nn.Linear(hidden_dims[0], output_dim)
        self.mid_out = nn.Linear(output_dim, hidden_dims[1])
        self.linears2 = ([nn.Linear(hidden_dims[1], hidden_dims[1]) for i in range(hidden_n[1] - 1)])
        self.output = nn.Linear(hidden_dims[1], output_dim * 2)

    def forward(self, x):
        y = x
        y = F.relu(self.input(y))
        for i in range(len(self.linears1)):
            y = F.relu(self.linears1[i](y))
        y = self.mid_in(y)
        z = y
        y = F.relu(self.mid_out(y))
        for i in range(len(self.linears2)):
            y = F.relu(self.linears2[i](y))
        y = self.output(y)
        y = y.reshape(list(x.size()[:-1]) + [2, self.output_dim])
        mean, log_scale = torch.unbind(y, dim=-2)
        scale = torch.exp(log_scale)
        return mean + scale * z

# how to compose transport maps

# base_dist = dist.Normal(torch.zeros(2), torch.ones(2))
# t1 = T.AffineAutoregressive(AutoRegressiveNN(2, [40]))
# t2 = T.AffineAutoregressive(AutoRegressiveNN(2, [40]))

# composed = T.ComposeTransformModule([t1,t2])
# transform_iaf_mog = dist.TransformedDistribution(base_dist, [composed])

# optimizer = torch.optim.Adam(composed.parameters(), lr=5e-3)
