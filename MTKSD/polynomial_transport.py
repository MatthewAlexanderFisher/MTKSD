import torch

import numpy as np


def theta_init(order):
    theta_1 = torch.zeros(order + 1)
    theta_1[1] = 1.

    n = order + 1.
    r = 2
    numer = torch.from_numpy(np.arange(1, (n + r))).prod()
    denom = torch.from_numpy(np.arange(1, (n + r - 2))).prod() * torch.from_numpy(np.arange(1, 3)).prod()
    d_theta_2 = (numer / denom).type(torch.LongTensor)
    theta_2 = torch.zeros(d_theta_2)
    theta_2[1] = 1.

    theta = torch.cat([theta_1, theta_2])
    theta.requires_grad = True

    return theta


def polynomial_transform(x, theta, order):
    N = x.size()[0]

    len_theta = len(theta)
    theta = theta.reshape(len_theta, 1)

    x_1 = x[:, 0].reshape(N, 1)
    x_2 = x[:, 1].reshape(N, 1)

    X_1 = []
    for i in range(order + 1):
        x_1_powers = x_1 ** i
        X_1.append(x_1_powers)

    X_1 = torch.cat(X_1, dim=1)

    seq = torch.from_numpy(np.arange(1, (order + 1)))
    seq = torch.cat([torch.tensor([0]), seq])

    combns = torch.combinations(seq, with_replacement=True)
    n_row = combns.size()[0]
    combns_1 = combns[:, 0].reshape(n_row, 1)
    combns_2 = combns[:, 1].reshape(n_row, 1)
    combns_rev = torch.cat([combns_2, combns_1], dim=1)
    neq_rows = (combns_1 != combns_2).reshape(n_row)
    combns_rev = combns_rev[neq_rows, :]

    combns = torch.cat([combns, combns_rev], dim=0)
    sum_crit = (combns.sum(dim=1) <= order)
    combns = combns[sum_crit, :]
    n_combns = combns.size()[0]

    for i in seq:
        combns_i = combns[combns[:, 0] == i, :]
        sorted_combns_i, _ = torch.sort(combns_i[:, 1])
        combns[combns[:, 0] == i, 1] = sorted_combns_i

    combns_array = []
    for i in seq:
        sub_tensor = combns[combns[:, 0] == i, :]
        combns_array.append(sub_tensor)

    combns = torch.cat(combns_array, dim=0)

    X_2 = []
    for i in range(n_combns):
        powers = combns[i, :]
        x_1_powers = x_1 ** (powers[0])
        x_2_powers = x_2 ** (powers[1])
        mult_vec = x_1_powers * x_2_powers
        X_2.append(mult_vec)

    X_2 = torch.cat(X_2, dim=1)

    n_col_X_1 = X_1.size()[1]
    n_col_X_2 = X_2.size()[1]

    zero_mat_1 = torch.zeros(N, n_col_X_2)
    zero_mat_2 = torch.zeros(N, n_col_X_1)
    X_1 = torch.cat((X_1, zero_mat_1), dim=1)
    X_2 = torch.cat((zero_mat_2, X_2), dim=1)

    X = torch.cat((X_1, X_2), dim=1).reshape(2 * N, len_theta)

    T_theta_mat = torch.mm(X, theta).reshape(N, 2)
    return T_theta_mat