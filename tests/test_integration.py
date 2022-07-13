from infrabayes.integration import gauss_hermite_quadrature, monte_carlo_expectation
import pytest
import torch as th
import torch.distributions as thd


@pytest.mark.parametrize("exponent", [1, 2, 3])
def test_gauss_hermite(exponent):
    th.manual_seed(0)

    mean = th.linspace(-1, 1, 10)
    std = th.linspace(0.1, 1, 10)
    mu = thd.Normal(mean, std)
    mc = monte_carlo_expectation(mu, lambda x: x ** exponent, n=5000_000)
    quad = gauss_hermite_quadrature(mu, lambda x: x ** exponent)
    assert th.allclose(mc, quad.float(), atol=0.01)
