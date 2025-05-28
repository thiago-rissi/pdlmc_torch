from typing import Any, Callable, Tuple
import torch
from torch.func import grad, vmap
from functools import partial
import torch.nn.functional as F
from tqdm import tqdm

ConstraintFn = Callable[[torch.Tensor | Any], Any]
PotentialFn = Callable[[torch.Tensor | Any], float]
ProjectionFn = Callable[[torch.Tensor | Any], Any]


def u(
    x: torch.Tensor, lmbda: float, nu: float, f: Callable, g: Callable, h: Callable
) -> torch.Tensor:
    """
    Compute the potential function u for the PD-LMC algorithm.
    """
    return (
        f(x)
        + torch.inner(lmbda, g(x))
        + torch.inner(nu, torch.tensor(h(x), device=x.device))
    )


def update_step(
    x: torch.Tensor, grad_u: torch.Tensor, step_size: float, normals: torch.Tensor
) -> torch.Tensor:
    """
    Update the state in each step.
    """
    sqrt_coeff = torch.sqrt(torch.tensor(2 * step_size, device=x.device))

    x = x - step_size * grad_u + sqrt_coeff * normals
    return x


def langevin_step_torch(
    state: torch.Tensor, grad_u: Callable, proj: Callable, step_size: float
) -> torch.Tensor:
    """
    Perform a single Langevin step on the given state.
    """
    x = state

    normals = torch.randn_like(x, device=x.device)

    g = grad_u(x)

    x = update_step(x=x, grad_u=g, step_size=step_size, normals=normals)

    x = proj(x)
    new_state = x
    return new_state


class PDLMCSolver:
    def __init__(
        self,
        f: PotentialFn,
        g: ConstraintFn,
        h: ConstraintFn,
        lmc_steps: int,
        burnin: int,
        step_size_x: float,
        step_size_lmbda: float,
        step_size_nu: float,
        proj: ProjectionFn = lambda x: x,
    ):
        self.f = f
        self.g = g
        self.h = h
        self.u = partial(u, f=f, g=g, h=h)
        self.grad_u = grad(self.u)
        self.langevin_step = partial(
            langevin_step_torch, proj=proj, step_size=step_size_x
        )
        self.lmc_steps = lmc_steps
        self.burnin = burnin
        self.step_size_x = step_size_x
        self.step_size_lmbda = step_size_lmbda
        self.step_size_nu = step_size_nu

    def pd_step(
        self,
        state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        iter_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a single PD-LMC step.
        """
        x, lmbda, nu = state
        grad_u_lang = partial(self.grad_u, lmbda=lmbda, nu=nu)

        xs = []
        for j in range(self.lmc_steps + self.burnin):
            x = self.langevin_step(x, grad_u=grad_u_lang)
            xs.append(x)

        afterburnin = torch.stack(xs[self.burnin :])
        last_iterate = afterburnin[-1]
        qlmbda = vmap(self.g)(afterburnin).mean(axis=0)
        qnu = self.h(afterburnin)

        new_state = (
            last_iterate,
            F.relu(lmbda + self.step_size_lmbda * qlmbda),
            nu + self.step_size_nu * qnu,
        )

        return new_state

    def run_chain(
        self,
        iterations: int,
        initial_x: torch.Tensor | Any,
        initial_lmbda: torch.Tensor | Any,
        initial_nu: torch.Tensor | Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run a PD-LMC chain with the given parameters.
        """
        state = initial_x, initial_lmbda, initial_nu

        x, lmbd, nu = [], [], []
        for i in tqdm(range(iterations)):
            state = self.pd_step(
                state,
                i,
            )

            x.append(state[0])
            lmbd.append(state[1])
            nu.append(state[2])

        x = torch.stack(x)
        lmbd = torch.stack(lmbd)
        nu = torch.stack(nu)

        return x, lmbd, nu
