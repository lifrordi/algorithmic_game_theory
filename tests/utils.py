import functools
from collections.abc import Callable, Generator

import numpy as np


def create_general_sum_game(
    max_size: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a random two-player general-sum game."""

    num_rows = rng.integers(2, max_size, None)
    num_cols = rng.integers(2, max_size, None)

    min_utility = rng.integers(-100, 75, None)
    max_utility = rng.integers(min_utility + 1, 100, None)
    row_matrix = rng.integers(min_utility, max_utility, (num_rows, num_cols))
    row_matrix = row_matrix + 1e-5 * rng.random((num_rows, num_cols))

    min_utility = rng.integers(-100, 75, None)
    max_utility = rng.integers(min_utility + 1, 100, None)
    col_matrix = rng.integers(min_utility, max_utility, (num_rows, num_cols))
    col_matrix = col_matrix + 1e-5 * rng.random((num_rows, num_cols))

    return row_matrix, col_matrix


def create_zero_sum_game(max_size: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random two-player zero-sum game."""

    return create_general_sum_game(max_size, rng)[0]


def create_game_with_dominated_strategies(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a game with a guaranteed sequence of strictly dominated strategies."""

    num_rows = rng.integers(3, 20, None)
    num_cols = rng.integers(3, 20, None)
    min_utility = rng.integers(-100, 0, None)
    max_utility = rng.integers(min_utility + 1, 100, None)

    # Determine the size of the unsolvable core matrix
    core_rows = rng.integers(1, num_rows, None)
    core_cols = rng.integers(1, num_cols, None)

    # Determine the random sequence of eliminations
    elimination_order = ['row'] * (num_rows - core_rows) + ['col'] * (num_cols - core_cols)
    rng.shuffle(elimination_order)

    # Generate the random "core" of the game, which is the final solution
    row_matrix = rng.integers(min_utility, max_utility + 1, (core_rows, core_cols))
    col_matrix = rng.integers(min_utility, max_utility + 1, (core_rows, core_cols))

    # Iteratively add the dominated strategies
    for player in elimination_order:
        current_rows, current_cols = row_matrix.shape

        if player == 'row':
            new_row_p1 = rng.integers(min_utility, max_utility + 1, (1, current_cols))
            new_row_p2 = rng.integers(min_utility, max_utility + 1, (1, current_cols))
            row_matrix = np.vstack([row_matrix, new_row_p1])
            col_matrix = np.vstack([col_matrix, new_row_p2])

            dominated_idx, dominating_idx = current_rows, rng.choice(current_rows)
            penalty_vector = rng.integers(1, 6, current_cols)
            row_matrix[dominated_idx, :] = row_matrix[dominating_idx, :] - penalty_vector

        else:
            new_col_p1 = rng.integers(min_utility, max_utility + 1, (current_rows, 1))
            new_col_p2 = rng.integers(min_utility, max_utility + 1, (current_rows, 1))
            row_matrix = np.hstack([row_matrix, new_col_p1])
            col_matrix = np.hstack([col_matrix, new_col_p2])

            dominated_idx, dominating_idx = current_cols, rng.choice(current_cols)
            penalty_vector = rng.integers(1, 6, current_rows)
            col_matrix[:, dominated_idx] = col_matrix[:, dominating_idx] - penalty_vector

    row_matrix = np.astype(row_matrix, np.float64)
    col_matrix = np.astype(col_matrix, np.float64)

    return row_matrix, col_matrix


def create_random_strategy(num_actions: int, rng: np.random.Generator) -> np.ndarray:
    def _create_random_pure_strategy(num_actions, rng):
        strategy = np.zeros(num_actions)
        index = rng.integers(0, num_actions, None)
        strategy[index] = 1.0

        return strategy

    def _create_random_mixed_strategy(num_actions, rng):
        strategy = rng.random(num_actions)
        strategy = strategy / np.sum(strategy)

        return strategy

    if rng.random() < 0.5:
        return _create_random_pure_strategy(num_actions, rng)
    else:
        return _create_random_mixed_strategy(num_actions, rng)


def create_random_support(num_actions: int, rng: np.random.Generator) -> np.ndarray:
    support_size = rng.integers(1, num_actions + 1, None)
    support = rng.choice(num_actions, support_size, replace=False)

    return support


def parameterize_classical_tests(zero_sum_only: bool, rng: np.random.Generator) -> Generator:
    games = {
        'prisoners_dilemma': (
            np.array([[-1, -3], [0, -2]], np.float64),
            np.array([[-1, 0], [-3, -2]], np.float64),
        ),
        'battle_of_the_sexes': (
            np.array([[2, 0], [0, 1]], np.float64),
            np.array([[1, 0], [0, 2]], np.float64),
        ),
        'game_of_chicken': (
            np.array([[0, -1], [1, -10]], np.float64),
            np.array([[0, 1], [-1, -10]], np.float64),
        ),
        'stag_hunt': (
            np.array([[4, 0], [3, 3]], np.float64),
            np.array([[4, 3], [0, 3]], np.float64),
        ),
        'harmony': (
            np.array([[3, 1], [2, 4]], np.float64),
            np.array([[3, 1], [2, 4]], np.float64),
        ),
        'coordination': (
            np.array([[1, 0], [0, 1]], np.float64),
            np.array([[1, 0], [0, 1]], np.float64),
        ),
        'shapley': (
            np.array([[2, 0, 1], [1, 2, 0], [0, 1, 2]], np.float64),
            np.array([[1, 0, 2], [2, 1, 0], [0, 2, 1]], np.float64),
        ),
        'rock_paper_scissors': (
            np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], np.float64),
            np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]], np.float64),
        ),
        'matching_pennies': (
            np.array([[1, -1], [-1, 1]], np.float64),
            np.array([[-1, 1], [1, -1]], np.float64),
        ),
    }

    for row_matrix, col_matrix in games.values():
        if zero_sum_only and not np.all(col_matrix == -row_matrix):
            continue

        row_strategy = create_random_strategy(row_matrix.shape[0], rng)
        col_strategy = create_random_strategy(col_matrix.shape[1], rng)

        yield row_matrix, col_matrix, row_strategy, col_strategy


def parameterize_random_general_sum_tests(max_size: int, rng: np.random.Generator) -> Generator:
    while True:
        row_matrix, col_matrix = create_general_sum_game(max_size, rng)

        row_strategy = create_random_strategy(row_matrix.shape[0], rng)
        col_strategy = create_random_strategy(col_matrix.shape[1], rng)

        yield row_matrix, col_matrix, row_strategy, col_strategy


def parameterize_random_zero_sum_tests(max_size: int, rng: np.random.Generator) -> Generator:
    while True:
        row_matrix = create_zero_sum_game(max_size, rng)

        row_strategy = create_random_strategy(row_matrix.shape[0], rng)
        col_strategy = create_random_strategy(row_matrix.shape[1], rng)

        yield row_matrix, -row_matrix, row_strategy, col_strategy


def parameterize_random_dominance_tests(rng: np.random.Generator) -> Generator:
    while True:
        row_matrix, col_matrix = create_game_with_dominated_strategies(rng)

        # Not needed for dominance testing, but included to keep the interface consistent
        row_strategy = create_random_strategy(row_matrix.shape[0], rng)
        col_strategy = create_random_strategy(col_matrix.shape[1], rng)

        yield row_matrix, col_matrix, row_strategy, col_strategy


def cache_data_stream_per_function(fixture_function: Callable) -> Callable:
    cache = {}

    @functools.wraps(fixture_function)
    def wrapper(request, rng):
        test_func_name = request.node.originalname

        if test_func_name not in cache:
            cache[test_func_name] = fixture_function(request, rng)

        return cache[test_func_name]

    return wrapper
