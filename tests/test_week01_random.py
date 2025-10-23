import hashlib
import sys
from collections.abc import Generator

sys.path.append('solutions')

import numpy as np
import pytest
from pytest import FixtureRequest
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture

import week01
from utils import (
    cache_data_stream_per_function,
    parameterize_random_dominance_tests,
    parameterize_random_general_sum_tests,
    parameterize_random_zero_sum_tests,
)

BASE_SEED = 101


@pytest.fixture
def rng(request: FixtureRequest) -> np.random.Generator:
    """Create a random seed based on the function name to ensure reproducibility."""

    digest = hashlib.md5(request.node.originalname.encode()).hexdigest()
    seed = (BASE_SEED + int(digest, 16)) % (2**32)

    return np.random.default_rng(seed)


@pytest.fixture
@cache_data_stream_per_function
def general_sum_data_stream(request: FixtureRequest, rng: np.random.Generator) -> Generator:
    return parameterize_random_general_sum_tests(100, rng)


@pytest.fixture
@cache_data_stream_per_function
def zero_sum_data_stream(request: FixtureRequest, rng: np.random.Generator) -> Generator:
    return parameterize_random_zero_sum_tests(100, rng)


@pytest.fixture
@cache_data_stream_per_function
def dominance_data_stream(request: FixtureRequest, rng: np.random.Generator) -> Generator:
    return parameterize_random_dominance_tests(rng)


@pytest.mark.parametrize('general_sum_data_stream', range(5), indirect=True)
def test_evaluate_general_sum(
    general_sum_data_stream: Generator,
    ndarrays_regression: NDArraysRegressionFixture,
    request: FixtureRequest,
) -> None:
    row_matrix, col_matrix, row_strategy, col_strategy = next(general_sum_data_stream)

    row_utility, col_utility = week01.evaluate_general_sum(
        row_matrix, col_matrix, row_strategy, col_strategy
    )

    assert row_utility.dtype == np.float64, 'Incorrect dtype!'
    assert col_utility.dtype == np.float64, 'Incorrect dtype!'

    ndarrays_regression.check(
        {'row_utility': row_utility, 'col_utility': col_utility},
        f'{request.node.originalname}{request.node.callspec.indices["general_sum_data_stream"]}',
    )


@pytest.mark.parametrize('zero_sum_data_stream', range(5), indirect=True)
def test_evaluate_zero_sum(
    zero_sum_data_stream: Generator,
    ndarrays_regression: NDArraysRegressionFixture,
    request: FixtureRequest,
) -> None:
    row_matrix, _, row_strategy, col_strategy = next(zero_sum_data_stream)

    row_utility, col_utility = week01.evaluate_zero_sum(row_matrix, row_strategy, col_strategy)

    assert row_utility.dtype == np.float64, 'Incorrect dtype!'
    assert col_utility.dtype == np.float64, 'Incorrect dtype!'

    ndarrays_regression.check(
        {'row_utility': row_utility, 'col_utility': col_utility},
        f'{request.node.originalname}{request.node.callspec.indices["zero_sum_data_stream"]}',
    )


@pytest.mark.parametrize('general_sum_data_stream', range(5), indirect=True)
def test_best_response_against_row(
    general_sum_data_stream: Generator,
    ndarrays_regression: NDArraysRegressionFixture,
    request: FixtureRequest,
) -> None:
    _, col_matrix, row_strategy, _ = next(general_sum_data_stream)

    col_strategy = week01.calculate_best_response_against_row(col_matrix, row_strategy)

    assert col_strategy.dtype == np.float64, 'Incorrect dtype!'

    ndarrays_regression.check(
        {'col_strategy': col_strategy},
        f'{request.node.originalname}{request.node.callspec.indices["general_sum_data_stream"]}',
    )


@pytest.mark.parametrize('general_sum_data_stream', range(5), indirect=True)
def test_best_response_against_col(
    general_sum_data_stream: Generator,
    ndarrays_regression: NDArraysRegressionFixture,
    request: FixtureRequest,
) -> None:
    row_matrix, _, _, col_strategy = next(general_sum_data_stream)

    row_strategy = week01.calculate_best_response_against_col(row_matrix, col_strategy)

    assert row_strategy.dtype == np.float64, 'Incorrect dtype!'

    ndarrays_regression.check(
        {'row_strategy': row_strategy},
        f'{request.node.originalname}{request.node.callspec.indices["general_sum_data_stream"]}',
    )


@pytest.mark.parametrize('general_sum_data_stream', range(5), indirect=True)
def test_evaluate_row_against_best_response(
    general_sum_data_stream: Generator,
    ndarrays_regression: NDArraysRegressionFixture,
    request: FixtureRequest,
) -> None:
    row_matrix, col_matrix, row_strategy, _ = next(general_sum_data_stream)

    row_utility = week01.evaluate_row_against_best_response(row_matrix, col_matrix, row_strategy)

    assert row_utility.dtype == np.float64, 'Incorrect dtype!'

    ndarrays_regression.check(
        {'row_utility': row_utility},
        f'{request.node.originalname}{request.node.callspec.indices["general_sum_data_stream"]}',
    )


@pytest.mark.parametrize('general_sum_data_stream', range(5), indirect=True)
def test_evaluate_col_against_best_response(
    general_sum_data_stream: Generator,
    ndarrays_regression: NDArraysRegressionFixture,
    request: FixtureRequest,
) -> None:
    row_matrix, col_matrix, _, col_strategy = next(general_sum_data_stream)

    col_utility = week01.evaluate_col_against_best_response(row_matrix, col_matrix, col_strategy)

    assert col_utility.dtype == np.float64, 'Incorrect dtype!'

    ndarrays_regression.check(
        {'col_utility': col_utility},
        f'{request.node.originalname}{request.node.callspec.indices["general_sum_data_stream"]}',
    )


@pytest.mark.parametrize('dominance_data_stream', range(5), indirect=True)
def test_find_strictly_dominated_actions(
    dominance_data_stream: Generator,
    ndarrays_regression: NDArraysRegressionFixture,
    request: FixtureRequest,
) -> None:
    row_matrix, col_matrix, *_ = next(dominance_data_stream)

    row_dominated_actions = week01.find_strictly_dominated_actions(row_matrix)
    col_dominated_actions = week01.find_strictly_dominated_actions(np.transpose(col_matrix))

    # Sort to ensure a deterministic order for regression testing
    row_dominated_actions = np.sort(row_dominated_actions)
    col_dominated_actions = np.sort(col_dominated_actions)

    assert row_dominated_actions.dtype == np.int64, 'Incorrect dtype!'
    assert col_dominated_actions.dtype == np.int64, 'Incorrect dtype!'

    ndarrays_regression.check(
        {
            'row_dominated_actions': row_dominated_actions,
            'col_dominated_actions': col_dominated_actions,
        },
        f'{request.node.originalname}{request.node.callspec.indices["dominance_data_stream"]}',
    )


@pytest.mark.parametrize('dominance_data_stream', range(5), indirect=True)
def test_iterated_removal_of_dominated_strategies(
    dominance_data_stream: Generator,
    ndarrays_regression: NDArraysRegressionFixture,
    request: FixtureRequest,
) -> None:
    row_matrix, col_matrix, *_ = next(dominance_data_stream)

    row_matrix, col_matrix, row_actions, col_actions = (
        week01.iterated_removal_of_dominated_strategies(row_matrix, col_matrix)
    )

    assert row_matrix.dtype == np.float64, 'Incorrect dtype!'
    assert col_matrix.dtype == np.float64, 'Incorrect dtype!'
    assert row_actions.dtype == np.int64, 'Incorrect dtype!'
    assert col_actions.dtype == np.int64, 'Incorrect dtype!'

    ndarrays_regression.check(
        {
            'row_matrix': row_matrix,
            'col_matrix': col_matrix,
            'row_actions': row_actions,
            'col_actions': col_actions,
        },
        f'{request.node.originalname}{request.node.callspec.indices["dominance_data_stream"]}',
    )
