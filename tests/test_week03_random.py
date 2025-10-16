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

    strategies = week03.fictitious_play(row_matrix, col_matrix, num_iters=1000, naive=False)
    final_exploitability = week03.compute_exploitability(row_matrix, col_matrix, *strategies[-1])

    ndarrays_regression.check(
        {'final_exploitability': final_exploitability},
        f'{request.node.originalname}{request.node.callspec.indices["zero_sum_data_stream"]}',
    )


@pytest.mark.parametrize('zero_sum_data_stream', range(5), indirect=True)
def test_fictitious_play_naive(
    zero_sum_data_stream: Generator,
    ndarrays_regression: NDArraysRegressionFixture,
    request: FixtureRequest,
) -> None:
    row_matrix, col_matrix, *_ = next(zero_sum_data_stream)

    strategies = week03.fictitious_play(row_matrix, col_matrix, num_iters=1000, naive=True)
    final_exploitability = week03.compute_exploitability(row_matrix, col_matrix, *strategies[-1])

    ndarrays_regression.check(
        {'final_exploitability': final_exploitability},
        f'{request.node.originalname}{request.node.callspec.indices["zero_sum_data_stream"]}',
    )
