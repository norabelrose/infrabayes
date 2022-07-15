from abc import ABC, abstractmethod
from typing import Callable, Union
from .sa_measure import SaMeasure
import torch as th
import torch.distributions as thd


class InfraDistribution(ABC):
    """
    An infradistribution is a convex set of sa-measures, or affine-transformed
    probability measures, that has a few special properties: nonemptiness, closure,
    upper-completion, positivity, weak bounded minimals, and normalization. See
    https://www.lesswrong.com/posts/YAa4qcMyoucRS2Ykr/basic-inframeasure-theory for
    more details.

    Alternatively, via Legendre-Fenchel duality, an infradistribution can be viewed
    as a concave, monotone, uniformly continuous functional h over C(X, [0, 1]) s.t.
    h(0) = 0 and h(1) = 1. That is, it can be viewed as a "gadget for taking expected
    values" of bounded cost/loss/utility functions.
    """

    @abstractmethod
    def __call__(self, f: Callable[[th.Tensor], th.Tensor]) -> th.Tensor:
        """Compute the expected value of a function wrt this infradistribution."""
    
    @abstractmethod
    def entropy(self) -> th.Tensor:
        """
        In "Less Basic Inframeasure Theory," this is shown to be equal to the maximum
        entropy of the sa-measures inside the infradistribution. This is only derived
        for measures over finite sets- just as Shannon entropy is technically only
        valid for discrete distributions- but it's probably fine to use for continuous
        distributions as well.
        """


class InfraPolytope(InfraDistribution):
    """
    An infradistribution represented by a polytope of sa-measures.

    An expectation wrt an infradistribution is the infimum of the expectation wrt all
    its sa-measures. Since expectation is a linear functional and infradistributions
    are convex, the infimum will always be found at the boundary of the set; in fact,
    it will always be a "minimal point" as defined in Basic Inframeasure Theory. We can
    exploit this fact to represent infradistributions only using their minimal points.
    `InfraPolytope` supports only those with finitely many minimal points.
    """

    def __init__(self, batched_measure: Union[thd.Distribution, SaMeasure]):
        """Construct an infradistribution from a batch of sa-measures."""

        # Automatically wrap in an sa-measure for convenience.
        if isinstance(batched_measure, thd.Distribution):
            batched_measure = SaMeasure(batched_measure)
        
        if not batched_measure.mu.batch_shape:
            raise ValueError("SaMeasure should have at least one batch dimension")

        self._batched_measure = batched_measure

    def __call__(self, f: Callable[[th.Tensor], th.Tensor]) -> th.Tensor:
        """
        Compute the expected value of a function wrt this infradistribution.
        """
        return self._batched_measure(f).min(dim=0).values
    
    def entropy(self) -> th.Tensor:
        return self._batched_measure.entropy().max(dim=0).values

    def __repr__(self):
        return f"InfraPolytope({self._batched_measure})"
