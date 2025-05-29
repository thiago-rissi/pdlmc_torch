"""
Sampling from a 2D truncated Gaussian
"""

import os
import sys
import time
import pathlib
import numpy as np
import torch
import torch.nn.functional as F

CWD = pathlib.Path(os.path.dirname(__file__)).parent
sys.path.append(os.path.join(CWD, ".."))
from pdlmc import PDLMCSolver

device = torch.device("cpu")

ITERATIONS = int(5e5)


# Constraints
@torch.compile
def f(x: torch.Tensor):
    """
    Potential function for a 2D Gaussian centered at (2, 2) with identity covariance.
    """
    center = 2 * torch.ones_like(x)
    return torch.inner(x - center, x - center) / 0.5


@torch.compile
def g(x: torch.Tensor):
    """Constraint function for a unit circle in 2D."""
    norm_sq = torch.inner(x, x)
    return F.relu(norm_sq - 1.0) - 0.001


@torch.compile
def ellipsoid_constraint(x: torch.Tensor):
    """Constraint function for an ellipsoid."""
    A_inv = torch.tensor([[4.0, 0.0], [0.0, 0.25]], device=x.device)
    quad_form = torch.sum(x @ A_inv * x, dim=-1)
    return F.relu(quad_form - 1.0) - 0.001


def main(args: list[str]):
    start = time.process_time()
    print("Starting PD-LMC sampling for 2D Gaussian...")
    solver = PDLMCSolver(
        f=f,
        g=ellipsoid_constraint,
        h=lambda _: 0,
        lmc_steps=1,
        burnin=0,
        step_size_x=1e-3,
        step_size_lmbda=2e-1,
        step_size_nu=0,
        proj=lambda x: x,
    )

    x, lmbda, nu = solver.run_chain(
        iterations=ITERATIONS,
        initial_x=torch.zeros(2, device=device),
        initial_lmbda=torch.tensor(0.0, device=device),
        initial_nu=torch.tensor(0.0, device=device),
    )
    print(f"PD-LMC finished processing in {time.process_time()-start} seconds.")

    pathlib.Path(CWD, "data/results").mkdir(parents=True, exist_ok=True)
    np.savez(
        os.path.join(CWD, "data/results/2d_gaussian"),
        pdlmc_x=x.cpu().numpy(),
        pdlmc_lambda=lmbda.cpu().numpy(),
        pdlmc_nu=nu.cpu().numpy(),
    )


if __name__ == "__main__":
    main(sys.argv[1:])
