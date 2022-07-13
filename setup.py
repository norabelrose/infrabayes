from setuptools import setup


setup(
    name="infrabayes",
    version="0.1.0",
    description="Partial implementation of Vanessa Kosoy's infra-Bayesian decision theory.",
    author="Nora Belrose",
    install_requires=["cvxpy", "numpy", "scipy", "torch", "matplotlib"],
)
