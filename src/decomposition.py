import torch
import numpy as np
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F

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
        eigvals, eigvecs = torch.linalg.eigh(cov_init)
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        self.mu = nn.Parameter(torch.tensor(mu_init, dtype=torch.float32))
        self.mu_init = torch.tensor(mu_init, dtype=torch.float32)
        self.log_l1 = nn.Parameter(torch.log(eigvals[0]))
        self.log_l2 = nn.Parameter(torch.log(eigvals[1]))  
        theta0 = torch.atan2(eigvecs[1, 0], eigvecs[0, 0])
        self.theta = nn.Parameter(theta0)

    @property
    def lambda1(self):
        return torch.exp(self.log_l1)

    @property
    def lambda2(self):
        return torch.exp(self.log_l2)

    @property
    def R(self):
        c = torch.cos(self.theta)
        s = torch.sin(self.theta)
        return torch.stack([
            torch.stack([c, -s]),
            torch.stack([s,  c])
        ])

    @property
    def cov(self):
        # covariance matrix = R diag(λ) Rᵀ
        L = torch.diag(torch.stack([self.lambda1, self.lambda2]))
        return self.R @ L @ self.R.t()

    @property
    def radii(self):
        # geometric radii of ellipse (sqrt of eigenvalues)
        return torch.sqrt(torch.stack([self.lambda1, self.lambda2]))

    @property
    def max_radius(self):
        # radius along major axis under Mahalanobis metric
        # 1 / sqrt(min eigenvalue)
        return 1.0 / torch.sqrt(torch.min(self.lambda1, self.lambda2))

    @property
    def bounding_radius(self):
        # radius of the smallest circle that contains the ellipse
        return torch.sqrt(torch.max(self.lambda1, self.lambda2))



def containment_penalty(child: 'OptimizableEllipse', parent: 'OptimizableEllipse', weight=1, eps=1e-6): 
    mu_c, mu_p = child.mu, parent.mu 
    Sigma_c, Sigma_p = child.cov, parent.cov
    A = torch.linalg.solve(Sigma_p, Sigma_c) 
    eigvals = torch.linalg.eigvalsh(A) 
    lambda_axes = torch.max(eigvals) 
    delta = mu_c - mu_p 
    lambda_shift = delta @ torch.linalg.solve(Sigma_p, delta) 
    lambda_star = lambda_axes + lambda_shift + eps 
    loss = weight * torch.relu(lambda_star - 1)**2 
    return loss


def minimal_movement_penalty(ellipse, weight=0.01):
    return weight * torch.norm(ellipse.mu - ellipse.mu_init)**2


def shrinkage_penalty(child, parent, alpha=0.8, weight=1.0):
    """
    Encourage child ellipse to be smaller than parent.
    """
    return weight * (
        torch.relu(child.lambda1 - alpha * parent.lambda1)**2 +
        torch.relu(child.lambda2 - alpha * parent.lambda2)**2
    )


def orientation_penalty(child, parent, weight=0.1):
    return weight * (child.theta - parent.theta)**2


def regularization_penalty(ellipse, weight=0.001):
    return weight * (ellipse.lambda1**2 + ellipse.lambda2**2)


def disjoint_penalty(E1, E2, eps=0.1, weight=10):
    mu1, mu2 = E1.mu, E2.mu
    r1 = E1.bounding_radius
    r2 = E2.bounding_radius
    dist = torch.norm(mu1 - mu2)
    overlap = 2*(r1 + r2 + eps) -dist
    return weight * torch.relu(overlap)**2


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
            loss += containment_penalty(E_c, E_p, weight = 10)
            loss += shrinkage_penalty(E_c, E_p, 1/len(children), weight=5)
    for e in ellipses.values():
        loss += minimal_movement_penalty(e, weight=0.001)
    for a, b in disjoint_pairs:
        loss += disjoint_penalty(ellipses[a], ellipses[b], weight = 10) 
    return loss

def optimize_hierarchy_adaptive(
    ellipse_dict,
    hierarchy,
    disjoint_pairs,
    lr=2e-3,
    alpha=0.8,
    target_loss=0.5,
    patience=50,         
    lr_factor=0.1,        
    min_lr=1e-4
):
    """
    ellipse_dict: dict of iri -> [mu_2d, cov_2d]
    hierarchy: dict of parent_iri -> list of child_iri
    """

    ellipses = {
        iri: OptimizableEllipse(mu, cov)
        for iri, (mu, cov) in ellipse_dict.items()
    }

    params = [p for e in ellipses.values() for p in e.parameters()]
    optimizer = Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_factor,
        patience=patience,
        min_lr=min_lr,
    )

    loss = calculate_loss(ellipses, hierarchy, disjoint_pairs, alpha)
    step = 0

    while loss > target_loss:
        optimizer.zero_grad()
        loss = calculate_loss(ellipses, hierarchy, disjoint_pairs, alpha)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        step += 1
        if step % 200 == 0:
            print(f"[step {step}] loss = {loss.item():.4f} | lr = {optimizer.param_groups[0]['lr']:.6f}")
    print(f"final loss = {loss:.4f} at iteration {step}")
    return ellipses
