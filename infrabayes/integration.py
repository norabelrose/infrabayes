"""Math routines for approximating integrals."""
import math

import numpy as np
import torch as th
import torch.distributions as thd
from functools import lru_cache
from typing import Callable


@lru_cache()
def gauss_hermite_params(n: int, device: th.device) -> tuple[th.Tensor, th.Tensor]:
    locs, weights = np.polynomial.hermite.hermgauss(n)
    locs = th.as_tensor(locs, device=device)
    weights = th.as_tensor(weights, device=device)
    return locs, weights


def gauss_hermite_quadrature(
    mu: thd.Normal, f: Callable[[th.Tensor], th.Tensor], n: int = 20
) -> th.Tensor:
    """
    Compute a close deterministic approximation to the expectation of a function over a
    Gaussian probability distribution using Gauss-Hermite quadrature.
    """
    sig_sq = mu.variance
    locs, weights = gauss_hermite_params(n, sig_sq.device)
    padded_locs = locs.view(*locs.shape, *([1] * sig_sq.dim()))

    shifted_locs = th.sqrt(2.0 * sig_sq) * padded_locs + mu.mean
    log_probs = f(shifted_locs)
    padded_weights = weights.view(*weights.shape, *([1] * (log_probs.dim() - 1)))

    res = (1 / math.sqrt(math.pi)) * (log_probs * padded_weights)
    return res.sum(tuple(range(locs.dim())))


def monte_carlo_expectation(
    mu: thd.Distribution, f: Callable[[th.Tensor], th.Tensor], n: int = 1000
) -> th.Tensor:
    """
    Approximate the expectation of a function over using Monte Carlo sampling.
    """
    samples = mu.sample([n])  # type: ignore
    return f(samples).mean(0)
