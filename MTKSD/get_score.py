from torch.autograd import grad

# uses pytorch autograd to automatically get score from a distribution

def get_score(x, dist):
    log_probs = dist.log_prob(x)
    grads = grad(log_probs.sum(), x, create_graph=True)[0]
    return grads
