from collections import namedtuple
from typing import Any, Callable, Tuple
import numpy as np
import torch
from torch.func import grad, vmap
from functools import partial
from torch.nn import ReLU
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map
import torch.nn.functional as F
from tqdm import tqdm

ConstraintFn = Callable[[torch.Tensor | Any], Any]
PotentialFn = Callable[[torch.Tensor | Any], float]
ProjectionFn = Callable[[torch.Tensor | Any], Any]


def pdlmc_run_chain_torch(
    device: torch.device,
    f: PotentialFn,
    g: ConstraintFn,
    h: ConstraintFn,
    iterations: int,
    lmc_steps: int,
    burnin: int,
    step_size_x: float,
    step_size_lmbda: float,
    step_size_nu: float,
    initial_x: torch.Tensor | Any,
    initial_lmbda: torch.Tensor | Any,
    initial_nu: torch.Tensor | Any,
    proj: ProjectionFn = lambda x: x,
) -> Tuple:

    def u(x, lmbda, nu):
        return (
            f(x)
            + torch.inner(lmbda, g(x))
            + torch.inner(nu, torch.tensor(h(x), device=x.device))
        )

    grad_u = grad(u)

    def pd_step(
        state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], iter_num: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, lmbda, nu = state
        grad_u_lang = partial(grad_u, lmbda=lmbda, nu=nu)

        step = partial(
            langevin_step_torch, grad_u=grad_u_lang, proj=proj, step_size=step_size_x
        )

        for j in range(lmc_steps + burnin):
            init_state = x
            x = step(init_state)

        qlmbda = vmap(g)(x).mean(axis=0)
        qnu = h(x)

        new_state = (
            x,
            F.relu(lmbda + step_size_lmbda * qlmbda),
            nu + step_size_nu * qnu,
        )

        return new_state

    state = initial_x, initial_lmbda, initial_nu

    x, lmbd, nu = [], [], []
    for i in tqdm(range(iterations)):
        state = pd_step(state, i)
        x.append(state[0])
        lmbd.append(state[1])
        nu.append(state[2])

    x = torch.stack(x)
    lmbd = torch.stack(lmbd)
    nu = torch.stack(nu)

    return x, lmbd, nu


def langevin_step_torch(state: torch.Tensor, grad_u, proj, step_size) -> Tuple:
    x = state

    normals = torch.randn_like(x, device=x.device)

    g = grad_u(x)
    sqrt_coeff = torch.sqrt(torch.tensor(2 * step_size, device=x.device))

    step_function = lambda p, g, r: p - step_size * g + sqrt_coeff * r
    x = step_function(x, g, normals)

    x = proj(x)
    new_state = x
    return new_state
