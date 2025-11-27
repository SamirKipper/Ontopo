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

def containment_penalty(child: 'OptimizableEllipse', parent: 'OptimizableEllipse', weight=50.0, eps=1e-6):
    mu_c, mu_p = child.mu, parent.mu

    r_c = torch.max(child.lambda1(), child.lambda2())
    r_p = torch.max(parent.lambda1(), parent.lambda2())

    center_dist = torch.norm(mu_c - mu_p)

    # enforce: center_dist + r_c <= r_p  <=>  center_dist + r_c - r_p <= 0
    penalty_val = center_dist + r_c - r_p + eps

    return weight * torch.relu(penalty_val)**2

def containment_penalty_old(child: 'OptimizableEllipse', parent: 'OptimizableEllipse', weight=50.0, eps=1e-6): 
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

def quad_matrix(mu, Sigma_inv):
    mu = mu.view(-1,1)
    A = Sigma_inv
    b = -Sigma_inv @ mu
    c = (mu.T @ Sigma_inv @ mu - 1).squeeze()
    
    M = torch.zeros(3,3, dtype=torch.float32)
    M[:2,:2] = A
    M[:2, 2] = b.squeeze()
    M[2, :2] = b.squeeze()
    M[2,2] = c
    return M


def slemma_distance(E1, E2, num_samples=20):
    """
    Stable S-lemma-based separation between ellipses.
    Returns the minimum value of F(lambda).
    """

    mu1, mu2 = E1.mu, E2.mu
    d = (mu1 - mu2).view(-1, 1)

    S1 = E1.cov()
    S2 = E2.cov()

    # Sample λ values evenly from [0,1]
    lambdas = torch.linspace(0, 1, num_samples)

    vals = []
    for lam in lambdas:
        Sigma = lam * S1 + (1 - lam) * S2
        Sigma_inv = torch.inverse(Sigma + 1e-6 * torch.eye(2))
        F = (d.T @ Sigma_inv @ d).squeeze()
        vals.append(F)

    vals = torch.stack(vals)
    return vals.min()  # minimum S-lemma value

def disjoint_penalty(E1, E2, weight=20.0, margin=0.0):
    # centers
    mu1, mu2 = E1.mu, E2.mu

    # treat ellipses as circles using max axis length (largest std dev)
    r1 = torch.max(E1.lambda1(), E1.lambda2())
    r2 = torch.max(E2.lambda1(), E2.lambda2())

    # center distance
    d = torch.norm(mu1 - mu2)

    # circles are disjoint if d >= r1 + r2 + margin
    target = r1 + r2 + margin

    return weight * torch.relu(target - d)**2


def parent_size_penalty(parent, weight=0.1):
    r_p = torch.max(parent.lambda1(), parent.lambda2())
    return weight * r_p**2

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
            loss += containment_penalty(E_c, E_p, weight = 500000)
            # loss += shrinkage_penalty(E_c, E_p, alpha)
            # loss += orientation_penalty(E_c, E_p)
    # for e in ellipses.values():
        # loss += regularization_penalty(e)
    for a, b in disjoint_pairs:
        loss += disjoint_penalty(ellipses[a], ellipses[b], weight = 1000)   
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

def optimize_hierarchy_adaptive(
    ellipse_dict,
    hierarchy,
    disjoint_pairs,
    lr=2e-3,
    alpha=0.8,
    target_loss=0.5,
    patience=50,          # plateau patience
    lr_factor=0.2,        # shrink LR by 50%
    min_lr=1e-6
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

    # Adaptive LR scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_factor,
        patience=patience,
        min_lr=min_lr,
        ## verbose=True
    )

    loss = calculate_loss(ellipses, hierarchy, disjoint_pairs, alpha)
    step = 0

    while loss > target_loss:
        optimizer.zero_grad()
        loss = calculate_loss(ellipses, hierarchy, disjoint_pairs, alpha)
        loss.backward()
        optimizer.step()

        # update LR based on loss
        scheduler.step(loss)

        step += 1
        if step % 200 == 0:
            print(f"[step {step}] loss = {loss.item():.4f} | lr = {optimizer.param_groups[0]['lr']:.6f}")

    return ellipses
