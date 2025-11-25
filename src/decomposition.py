import torch
import numpy as np
from torch import nn
from torch.optim import Adam


# ======================================================
# Utility functions
# ======================================================

def cov_from_params(lambda1, lambda2, theta):
    R = torch.tensor([
        [torch.cos(theta), -torch.sin(theta)],
        [torch.sin(theta),  torch.cos(theta)]
    ], dtype=torch.float32)
    
    L = torch.diag(torch.stack([lambda1, lambda2]))
    return R @ L @ R.T


def mahalanobis_distance(x, mu, cov_inv):
    d = x - mu
    return d @ cov_inv @ d


# ======================================================
# Ellipse Parameter Module
# ======================================================

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


# ======================================================
# Hierarchy constraint losses
# ======================================================

def containment_penalty(child: OptimizableEllipse, parent: OptimizableEllipse, weight=1.0):
    """
    Soft penalty if the child ellipse exceeds parent boundary.
    """
    mu_c = child.mu
    mu_p = parent.mu

    cov_p = parent.cov()
    covinv_p = torch.linalg.inv(cov_p)
    d = mahalanobis_distance(mu_c, mu_p, covinv_p)
    A = covinv_p @ child.cov()
    eigvals = torch.linalg.eigvals(A).real
    r_child = torch.sqrt(torch.max(eigvals))

    return weight * torch.relu(d + r_child - 1.0)**2


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
    """
    Optional: prevents ellipses from blowing up or collapsing.
    """
    return weight * (ellipse.lambda1()**2 + ellipse.lambda2()**2)


def disjoint_penalty_slemma(E1, E2, margin=1.0):
    """
    Uses an S-lemma-inspired differentiable penalty to encourage disjoint ellipses.
    """
    mu1 = E1.mu
    mu2 = E2.mu
    n = mu1.shape[0]
    
    A1 = torch.inverse(E1.cov() + 1e-6*torch.eye(n))
    A2 = torch.inverse(E2.cov() + 1e-6*torch.eye(n))
    
    lam = torch.tensor(1.0, requires_grad=True)
    
    # Quadratic matrix
    Q11 = A1 + lam * A2  # (n,n)
    Q12 = -(A1 @ mu1 + lam * A2 @ mu2)  # (n,)
    Q22 = (mu1.T @ A1 @ mu1 + lam * mu2.T @ A2 @ mu2 - (1 + lam))  # scalar
    
    # Unsqueeze Q12 and Q22 to make proper 2D blocks
    Q12 = Q12.unsqueeze(1)           # (n,1)
    Q22 = Q22.unsqueeze(0).unsqueeze(1)  # (1,1)
    
    # Concatenate blocks
    top = torch.cat([Q11, Q12], dim=1)   # (n, n+1)
    bottom = torch.cat([Q12.T, Q22], dim=1)  # (1, n+1)
    Q = torch.cat([top, bottom], dim=0)      # (n+1, n+1)
    
    # Penalty: negative eigenvalues indicate violation
    eigvals = torch.linalg.eigvalsh(Q)
    penalty = torch.relu(margin - torch.min(eigvals))
    return penalty


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
        loss += disjoint_penalty_slemma(ellipses[a], ellipses[b], margin=1.0)   
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

