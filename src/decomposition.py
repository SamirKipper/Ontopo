import torch
import numpy as np
from torch import nn
from torch.optim import Adam


def cov_from_params(lambda1, lambda2, theta):
    R = torch.tensor([
        [torch.cos(theta), -torch.sin(theta)],
        [torch.sin(theta),  torch.cos(theta)]
    ], dtype=torch.float32)
    
    L = torch.diag(torch.stack([lambda1, lambda2]))
    return R @ L @ R.T


class OptimizableEllipse(nn.Module):
    def __init__(self, mu_init, cov_init):
        super().__init__()

        cov_init = torch.tensor(cov_init, dtype=torch.float32)

        # Extract eigenstructure
        eigvals, eigvecs = torch.linalg.eigh(cov_init)
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Params
        self.mu = nn.Parameter(torch.tensor(mu_init, dtype=torch.float32))

        self.log_l1 = nn.Parameter(torch.log(eigvals[0]))
        self.log_l2 = nn.Parameter(torch.log(eigvals[1]))

        theta0 = torch.atan2(eigvecs[1, 0], eigvecs[0, 0])
        self.theta = nn.Parameter(theta0)

    def lambda1(self):
        return torch.exp(self.log_l1)

    def lambda2(self):
        return torch.exp(self.log_l2)

    def cov(self):
        return cov_from_params(
            self.lambda1(),
            self.lambda2(),
            self.theta
        )


def containment_penalty(child: 'OptimizableEllipse', parent: 'OptimizableEllipse', weight=5.0, eps=1e-6):
    mu_c, mu_p = child.mu, parent.mu
    Sigma_c, Sigma_p = child.cov(), parent.cov()

    A = torch.linalg.solve(Sigma_p, Sigma_c)  
    eigvals = torch.linalg.eigvalsh(A)  
    lambda_axes = torch.max(eigvals)

    delta = mu_c - mu_p
    lambda_shift = delta @ torch.linalg.solve(Sigma_p, delta) 

    lambda_star = lambda_axes + lambda_shift + eps
    loss = weight * torch.relu(lambda_star - 1)**2
    return loss




def shrinkage_penalty(child, parent, alpha=0.8, weight=1.0):
    """
    Encourage child ellipse to be smaller than parent.
    """
    return weight * (
        torch.relu(child.lambda1() - alpha * parent.lambda1())**2 +
        torch.relu(child.lambda2() - alpha * parent.lambda2())**2
    )


def orientation_penalty(child, parent, weight=0.1):
    return weight * (child.theta - parent.theta)**2


def regularization_penalty(ellipse, weight=0.001):
    return weight * (ellipse.lambda1()**2 + ellipse.lambda2()**2)


def disjoint_penalty(E1, E2, iters=20, lr=0.05):
    """
    Returns signed minimal distance between two ellipsoids.
    Positive => disjoint
    Negative => overlapping (distance is penetration depth).
    """

    mu1, mu2 = E1.mu, E2.mu
    L1 = torch.linalg.cholesky(E1.cov() + 1e-6 * torch.eye(mu1.shape[0]))
    L2 = torch.linalg.cholesky(E2.cov() + 1e-6 * torch.eye(mu2.shape[0]))

    n = mu1.shape[0]

    # Parameter vectors on unit ball
    u1 = torch.zeros(n, requires_grad=True)
    u2 = torch.zeros(n, requires_grad=True)

    opt = torch.optim.SGD([u1, u2], lr=lr)

    for _ in range(iters):
        opt.zero_grad()

        # Project u onto unit ball (differentiable soft projection)
        v1 = u1 / (u1.norm() + 1e-8)
        v2 = u2 / (u2.norm() + 1e-8)

        x = mu1 + L1 @ v1
        y = mu2 + L2 @ v2

        d = torch.norm(x - y)
        d.backward(retain_graph=True)
        opt.step()

    # final distance
    return torch.relu(d.detach())




def calculate_loss(ellipses, hierarchy, disjoint_pairs, alpha):
    loss = 0.0
    for parent, children in hierarchy.items():
        if parent not in ellipses:           
            continue
        E_p = ellipses[parent]
        for c in children:
            if c not in ellipses:                   
                continue
            E_c = ellipses[c]
            loss += containment_penalty(E_c, E_p)
            loss += shrinkage_penalty(E_c, E_p, alpha)
            loss += orientation_penalty(E_c, E_p)
    for e in ellipses.values():
        loss += regularization_penalty(e)
    for a, b in disjoint_pairs:
        loss += disjoint_penalty(ellipses[a], ellipses[b])   
    return loss

def optimize_hierarchy(ellipse_dict, hierarchy, disjoint_pairs,  lr=2e-3, alpha=0.8, target_loss = 0.5):
    """
    ellipse_dict: dict of iri -> [mu_2d (numpy array), cov_2d (numpy 2x2 array)]
        Example:
        {
            "http://purl.org/x" : [mu, cov],
            ...
        }

    hierarchy: dict of parent_iri -> list of child_iri

    Returns:
        dict of iri -> OptimizableEllipse instance
    """
    ellipses = {}
    for iri, (mu, cov) in ellipse_dict.items():
        ellipses[iri] = OptimizableEllipse(mu, cov)
    optimizer = Adam(
        [p for e in ellipses.values() for p in e.parameters()],
        lr=lr
        )
    loss = calculate_loss(ellipses, hierarchy, disjoint_pairs, alpha)
    step = 0
    while loss > target_loss:
        optimizer.zero_grad()
        loss = calculate_loss(ellipses, hierarchy, disjoint_pairs, alpha)
        loss.backward()
        optimizer.step()
        step += 1
        if step % 200 == 0:
            print(f"[step {step}] loss = {loss.item():.4f}")
    return ellipses

