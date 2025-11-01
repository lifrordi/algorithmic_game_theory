import hashlib
import sys
from collections.abc import Generator

sys.path.append('solutions')

import numpy as np
import pytest
from pytest import FixtureRequest
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture

import week03
from utils import (
    cache_data_stream_per_function,
    parameterize_random_general_sum_tests,
    parameterize_random_zero_sum_tests,
)

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
    return parameterize_random_general_sum_tests(8, rng)


@pytest.fixture
@cache_data_stream_per_function
def zero_sum_data_stream(request: FixtureRequest, rng: np.random.Generator) -> Generator:
    return parameterize_random_zero_sum_tests(8, rng)


@pytest.mark.parametrize('general_sum_data_stream', range(5), indirect=True)
def test_compute_deltas(
    general_sum_data_stream: Generator,
    ndarrays_regression: NDArraysRegressionFixture,
    request: FixtureRequest,
) -> None:
    row_matrix, col_matrix, row_strategy, col_strategy = next(general_sum_data_stream)

    row_delta, col_delta = week03.compute_deltas(row_matrix, col_matrix, row_strategy, col_strategy)

    assert row_delta.dtype == np.float64, 'Incorrect dtype!'
    assert col_delta.dtype == np.float64, 'Incorrect dtype!'

    ndarrays_regression.check(
        {'row_delta': row_delta, 'col_delta': col_delta},
        f'{request.node.originalname}{request.node.callspec.indices["general_sum_data_stream"]}',
    )


@pytest.mark.parametrize('general_sum_data_stream', range(5), indirect=True)
def test_compute_nash_conv(
    general_sum_data_stream: Generator,
    ndarrays_regression: NDArraysRegressionFixture,
    request: FixtureRequest,
) -> None:
    row_matrix, col_matrix, row_strategy, col_strategy = next(general_sum_data_stream)

    nash_conv = week03.compute_nash_conv(row_matrix, col_matrix, row_strategy, col_strategy)

    assert nash_conv.dtype == np.float64, 'Incorrect dtype!'

    ndarrays_regression.check(
        {'nash_conv': nash_conv},
        f'{request.node.originalname}{request.node.callspec.indices["general_sum_data_stream"]}',
    )


@pytest.mark.parametrize('general_sum_data_stream', range(5), indirect=True)
def test_compute_exploitability(
    general_sum_data_stream: Generator,
    ndarrays_regression: NDArraysRegressionFixture,
    request: FixtureRequest,
) -> None:
    row_matrix, col_matrix, row_strategy, col_strategy = next(general_sum_data_stream)

    exploitability = week03.compute_exploitability(
        row_matrix, col_matrix, row_strategy, col_strategy
    )

    assert exploitability.dtype == np.float64, 'Incorrect dtype!'

    ndarrays_regression.check(
        {'exploitability': exploitability},
        f'{request.node.originalname}{request.node.callspec.indices["general_sum_data_stream"]}',
    )


@pytest.mark.parametrize('zero_sum_data_stream', range(5), indirect=True)
def test_fictitious_play(
    zero_sum_data_stream: Generator,
    ndarrays_regression: NDArraysRegressionFixture,
    request: FixtureRequest,
) -> None:
    row_matrix, col_matrix, *_ = next(zero_sum_data_stream)

    strategies = week03.fictitious_play(row_matrix, col_matrix, num_iters=5, naive=False)
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
        f'{request.node.originalname}{request.node.callspec.indices["zero_sum_data_stream"]}',
    )


@pytest.mark.parametrize('zero_sum_data_stream', range(5), indirect=True)
def test_fictitious_play_naive(
    zero_sum_data_stream: Generator,
    ndarrays_regression: NDArraysRegressionFixture,
    request: FixtureRequest,
) -> None:
    row_matrix, col_matrix, *_ = next(zero_sum_data_stream)

    strategies = week03.fictitious_play(row_matrix, col_matrix, num_iters=5, naive=True)
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
        f'{request.node.originalname}{request.node.callspec.indices["zero_sum_data_stream"]}',
    )
