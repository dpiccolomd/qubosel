import numpy as np
import pytest

from qubosel.qubo import QUBO


def random_qubo(n: int, rng: np.random.Generator, scale: float = 1.0) -> QUBO:
    A = rng.normal(scale=scale, size=(n, n))
    return QUBO(0.5 * (A + A.T))


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def small_qubo(rng):
    return random_qubo(8, rng)
