from abc import ABC, abstractmethod
from typing import Callable
from .sa_measure import SaMeasure
import torch as th


class InfraDistribution(ABC):
    """
    An infradistribution is a convex set of sa-measures, or affine-transformed
    probability measures, that has a few special properties: nonemptiness, closure,
    upper-completion, positivity, weak bounded minimals, and normalization. See
    https://www.lesswrong.com/posts/YAa4qcMyoucRS2Ykr/basic-inframeasure-theory for
    more details.

    An expectation wrt an infradistribution is the infimum of the expectation wrt all
    its sa-measures. Since expectation is a linear functional and infradistributions
    are convex, the infimum will always be found at the boundary of the set; in fact,
    it will always be a "minimal point" as defined in Basic Inframeasure Theory. We can
    exploit this fact to represent infradistributions only using their minimal points.
    For simplicity and computational efficiency, we only support those with finitely
    many minimal points; that is, polytopes.

    As the dimensionality increases, however, there may be exponentially many minimal
    points even if the polytope only has linearly many facets, so it can be more
    efficient to represent the infradistribution using linear inequalities.
    """

    @abstractmethod
    def __call__(self, f: Callable[[th.Tensor], th.Tensor]) -> th.Tensor:
        """Compute the expected value of a function wrt this infradistribution."""


class InfraPolytope(InfraDistribution):
    """An infradistribution represented by a polytope of sa-measures."""

    def __init__(self, batched_measure: SaMeasure, dim: int = 0):
        """Construct an infradistribution from a batch of sa-measures."""

        num_batch_dims = len(batched_measure.mu.batch_shape)
        if num_batch_dims <= dim:
            raise ValueError(
                f"Expected an sa-measure with at least {dim + 1} batch dimension(s); got {num_batch_dims}"
            )

        self._batched_measure = batched_measure
        self._dim = dim

    def __call__(self, f: Callable[[th.Tensor], th.Tensor]) -> th.Tensor:
        """
        Compute the expected value of a function wrt this infradistribution.
        """
        return self._batched_measure(f).min(dim=self._dim)

    def __repr__(self):
        return f"InfraPolytope({self._batched_measure})"
