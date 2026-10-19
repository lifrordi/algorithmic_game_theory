import hashlib
import sys
from collections.abc import Generator

sys.path.append('solutions')

import numpy as np
import pytest
from pytest import FixtureRequest
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture

import week02
from utils import (
    cache_data_stream_per_function,
    check_support_strategy,
    create_random_support,
    parameterize_random_general_sum_tests,
    parameterize_random_zero_sum_tests,
)

BASE_SEED = 181


@pytest.fixture
def rng(request: FixtureRequest) -> np.random.Generator:
    """Create a random seed based on the function name to ensure reproducibility."""

    digest = hashlib.md5(request.node.originalname.encode()).hexdigest()
    seed = (BASE_SEED + int(digest, 16)) % (2**32)

    return np.random.default_rng(seed)


@pytest.fixture
@cache_data_stream_per_function
def support_rng(request: FixtureRequest, rng: np.random.Generator) -> np.random.Generator:
    """Share a single generator across all parametrizations of a test to vary the supports."""

    return rng


@pytest.fixture
@cache_data_stream_per_function
def general_sum_data_stream(request: FixtureRequest, rng: np.random.Generator) -> Generator:
    return parameterize_random_general_sum_tests(8, rng)


@pytest.fixture
@cache_data_stream_per_function
def zero_sum_data_stream(request: FixtureRequest, rng: np.random.Generator) -> Generator:
    return parameterize_random_zero_sum_tests(8, rng)


@pytest.mark.parametrize('general_sum_data_stream', range(5), indirect=True)
def test_verify_support_general_sum(
    general_sum_data_stream: Generator,
    ndarrays_regression: NDArraysRegressionFixture,
    request: FixtureRequest,
    support_rng: np.random.Generator,
) -> None:
    row_matrix, col_matrix, *_ = next(general_sum_data_stream)
    row_support = create_random_support(row_matrix.shape[0], support_rng)
    col_support = create_random_support(col_matrix.shape[1], support_rng)

    col_probs = week02.verify_support(row_matrix, row_support, col_support)
    row_probs = week02.verify_support(np.transpose(col_matrix), col_support, row_support)

    # The probabilities are not unique in general, so we check their properties instead
    if col_probs is not None:
        check_support_strategy(row_matrix, row_support, col_support, col_probs)

    if row_probs is not None:
        check_support_strategy(np.transpose(col_matrix), col_support, row_support, row_probs)

    ndarrays_regression.check(
        {
            'row_probs_exist': np.array(row_probs is not None, np.int64),
            'col_probs_exist': np.array(col_probs is not None, np.int64),
        },
        f'{request.node.originalname}{request.node.callspec.indices["general_sum_data_stream"]}',
    )


@pytest.mark.parametrize('zero_sum_data_stream', range(5), indirect=True)
def test_verify_support_zero_sum(
    zero_sum_data_stream: Generator,
    ndarrays_regression: NDArraysRegressionFixture,
    request: FixtureRequest,
    support_rng: np.random.Generator,
) -> None:
    row_matrix, col_matrix, *_ = next(zero_sum_data_stream)
    row_support = create_random_support(row_matrix.shape[0], support_rng)
    col_support = create_random_support(col_matrix.shape[1], support_rng)

    col_probs = week02.verify_support(row_matrix, row_support, col_support)
    row_probs = week02.verify_support(np.transpose(col_matrix), col_support, row_support)

    # The probabilities are not unique in general, so we check their properties instead
    if col_probs is not None:
        check_support_strategy(row_matrix, row_support, col_support, col_probs)

    if row_probs is not None:
        check_support_strategy(np.transpose(col_matrix), col_support, row_support, row_probs)

    ndarrays_regression.check(
        {
            'row_probs_exist': np.array(row_probs is not None, np.int64),
            'col_probs_exist': np.array(col_probs is not None, np.int64),
        },
        f'{request.node.originalname}{request.node.callspec.indices["zero_sum_data_stream"]}',
    )


@pytest.mark.parametrize('general_sum_data_stream', range(5), indirect=True)
def test_support_enumeration_general_sum(
    general_sum_data_stream: Generator,
    ndarrays_regression: NDArraysRegressionFixture,
    request: FixtureRequest,
) -> None:
    row_matrix, col_matrix, *_ = next(general_sum_data_stream)

    equilibria = week02.support_enumeration(row_matrix, col_matrix)
    equilibria = sorted(equilibria, key=lambda x: (x[0].tobytes(), x[1].tobytes()))

    for row_strategy, col_strategy in equilibria:
        assert row_strategy.dtype == np.float64, 'Incorrect dtype!'
        assert col_strategy.dtype == np.float64, 'Incorrect dtype!'

    ndarrays_regression.check(
        {f'{i}': np.concatenate(x) for i, x in enumerate(equilibria)},
        f'{request.node.originalname}{request.node.callspec.indices["general_sum_data_stream"]}',
    )


@pytest.mark.parametrize('zero_sum_data_stream', range(5), indirect=True)
def test_support_enumeration_zero_sum(
    zero_sum_data_stream: Generator,
    ndarrays_regression: NDArraysRegressionFixture,
    request: FixtureRequest,
) -> None:
    row_matrix, col_matrix, *_ = next(zero_sum_data_stream)

    equilibria = week02.support_enumeration(row_matrix, col_matrix)
    equilibria = sorted(equilibria, key=lambda x: (x[0].tobytes(), x[1].tobytes()))

    for row_strategy, col_strategy in equilibria:
        assert row_strategy.dtype == np.float64, 'Incorrect dtype!'
        assert col_strategy.dtype == np.float64, 'Incorrect dtype!'

    ndarrays_regression.check(
        {f'{i}': np.concatenate(x) for i, x in enumerate(equilibria)},
        f'{request.node.originalname}{request.node.callspec.indices["zero_sum_data_stream"]}',
    )
