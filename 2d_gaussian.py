"""
Sampling from a 2D truncated Gaussian
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

CWD = os.path.dirname(__file__)
sys.path.append(os.path.join(CWD, ".."))
from pdlmc import pdlmc_run_chain_torch as pdlmc_run_chain

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ITERATIONS = int(2.5e6)


# Constraints
def f(x):
    x = x.to(device)
    target = 2 * torch.ones_like(x)
    return torch.inner(x - target, x - target) / 0.5


def g(x):
    x = x.to(device)
    norm_sq = torch.inner(x, x)
    return F.relu(norm_sq - 1.0) - 0.001


# PD-LMC
start = time.process_time()
x, lmbda, nu = pdlmc_run_chain(
    device=device,
    f=f,
    g=g,
    h=lambda _: 0,
    iterations=ITERATIONS,
    lmc_steps=1,
    burnin=0,
    step_size_x=1e-3,
    step_size_lmbda=2e-1,
    step_size_nu=0,
    initial_x=torch.zeros(2, device=device),
    initial_lmbda=torch.tensor(0.0, device=device),
    initial_nu=torch.tensor(0.0, device=device),
)
print(f"PD-LMC finished processing in {time.process_time()-start} seconds.")

np.savez(
    os.path.join(CWD, "2d_gaussian"),
    pdlmc_x=x.cpu().numpy(),
    pdlmc_lambda=lmbda.cpu().numpy(),
    pdlmc_nu=nu.cpu().numpy(),
)
