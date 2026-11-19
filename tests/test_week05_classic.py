import hashlib
import sys
from collections.abc import Generator

sys.path.append('solutions')

import numpy as np
import pytest
from pytest import FixtureRequest
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture

import week03
import week05
from utils import cache_data_stream_per_function, parameterize_classical_tests

NUM_GAMES = 9
NUM_ZERO_SUM_GAMES = 2
BASE_SEED = 141


@pytest.fixture
def rng(request: FixtureRequest) -> np.random.Generator:
    """Create a random seed based on the function name to ensure reproducibility."""

    digest = hashlib.md5(request.node.originalname.encode()).hexdigest()
    seed = (BASE_SEED + int(digest, 16)) % (2**32)

    return np.random.default_rng(seed)


@pytest.fixture
@cache_data_stream_per_function
def general_sum_data_stream(request: FixtureRequest, rng: np.random.Generator) -> Generator:
    return parameterize_classical_tests(False, rng)


@pytest.fixture
@cache_data_stream_per_function
def zero_sum_data_stream(request: FixtureRequest, rng: np.random.Generator) -> Generator:
    return parameterize_classical_tests(True, rng)


@pytest.mark.parametrize(
    ('regrets', 'expected_strategy'),
    [
        pytest.param([1.0, 2.0, 3.0], [1 / 6, 2 / 6, 3 / 6], id='all_positive'),
        pytest.param([2.0, -1.0, 6.0], [0.25, 0.0, 0.75], id='mixed_signs'),
        pytest.param([0.0, 5.0, -5.0], [0.0, 1.0, 0.0], id='single_positive'),
        pytest.param([1e-3, 3e-3], [0.25, 0.75], id='small_values'),
        pytest.param([-1.0, -2.0, -3.0], [1 / 3, 1 / 3, 1 / 3], id='all_negative'),
        pytest.param([-1.0, 0.0], [0.5, 0.5], id='non_positive'),
        pytest.param([0.0, 0.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25], id='all_zero'),
        pytest.param([4.0], [1.0], id='single_action'),
    ],
)
def test_regret_matching(regrets: list[float], expected_strategy: list[float]) -> None:
    regrets = np.array(regrets, dtype=np.float64)
    regrets_copy = regrets.copy()

    strategy = week05.regret_matching(regrets)

    assert strategy.dtype == np.float64, 'Incorrect dtype!'
    assert strategy.shape == regrets.shape, 'Incorrect shape!'
    assert np.array_equal(regrets, regrets_copy), 'The input regrets were modified!'
    assert np.allclose(strategy, expected_strategy), 'Incorrect strategy!'


@pytest.mark.parametrize('general_sum_data_stream', range(NUM_GAMES), indirect=True)
def test_regret_minimization(
    general_sum_data_stream: Generator,
    ndarrays_regression: NDArraysRegressionFixture,
    request: FixtureRequest,
) -> None:
    row_matrix, col_matrix, *_ = next(general_sum_data_stream)

    strategies = week05.regret_minimization(row_matrix, col_matrix, num_iters=5)
    exploitability = week03.compute_exploitability(row_matrix, col_matrix, *strategies[-1])

    assert len(strategies) == 5, 'Incorrect number of strategies returned!'

    for row_strategy, col_strategy in strategies:
        assert row_strategy.dtype == np.float64, 'Incorrect dtype!'
        assert col_strategy.dtype == np.float64, 'Incorrect dtype!'

        assert np.isclose(np.sum(row_strategy), 1.0), 'Strategy does not sum to 1!'
        assert np.isclose(np.sum(col_strategy), 1.0), 'Strategy does not sum to 1!'

    assert exploitability.dtype == np.float64, 'Incorrect dtype!'

    ndarrays_regression.check(
        {
            'row_strategy': np.round(strategies[-1][0], 8),
            'col_strategy': np.round(strategies[-1][1], 8),
            'exploitability': np.round(exploitability, 8),
        },
        f'{request.node.originalname}{request.node.callspec.indices["general_sum_data_stream"]}',
    )
