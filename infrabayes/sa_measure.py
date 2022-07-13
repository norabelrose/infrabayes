from dataclasses import dataclass
from .integration import gauss_hermite_quadrature
from typing import Callable, Optional, Union
import torch as th
import torch.distributions as thd


@dataclass
class SaMeasure:
    # The underlying probability measure.
    mu: thd.Distribution
    scale: Optional[th.Tensor] = None
    bias: Optional[th.Tensor] = None

    # Support shorthand naming convention used in the LW sequence.
    λ = scale
    b = bias

    def __add__(self, other: Union[float, "SaMeasure", th.Tensor]) -> "SaMeasure":
        """
        Addition is overloaded to represent two different things, depending on the type
        of the `other` argument:
        - If `other` is an `SaMeasure`, the result is the sum of the two sa-measures.
        - If `other` is a `Tensor`, the result is a copy of `self` whose bias term has
        `other` added to it.
        """
        if isinstance(other, (float, th.Tensor)):
            new_bias = self.bias + other if self.bias is not None else other
            return SaMeasure(self.mu, self.scale, th.as_tensor(new_bias))

        elif isinstance(other, SaMeasure):
            raise NotImplementedError(
                "Addition of sa-measures is not yet implemented- consider submitting a PR!"
            )
        else:
            raise TypeError(f"Cannot add {type(other)} to SaMeasure")

    def __call__(self, f: Callable[[th.Tensor], th.Tensor]) -> th.Tensor:
        """
        Compute the expected value of a function wrt this sa-measure.
        """
        if isinstance(self.mu, thd.Normal):
            E = gauss_hermite_quadrature(self.mu, f)
            if self.scale is not None:
                E = E * self.scale
            if self.bias is not None:
                E = E + self.bias
            return E
        else:
            raise NotImplementedError(
                f"Expected values not yet implemented for {type(self.mu)}"
            )

    def __mul__(self, scalar: Union[float, th.Tensor]) -> "SaMeasure":
        """Scalar multiplication of sa-measures."""
        new_scale = self.scale * scalar if self.scale is not None else None
        new_bias = self.bias * scalar if self.bias is not None else None
        return SaMeasure(self.mu, new_scale, new_bias)
