from collections.abc import Generator

import numpy as np


def create_general_sum_game(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generate a random two-player general-sum game."""

    num_rows = rng.integers(2, 100, None, np.int32)
    num_cols = rng.integers(2, 100, None, np.int32)

    min_utility = rng.integers(-100, 75, None, np.int32)
    max_utility = rng.integers(min_utility + 1, 100, None, np.int32)
    row_matrix = rng.integers(min_utility, max_utility, (num_rows, num_cols), np.int32)

    min_utility = rng.integers(-100, 75, None, np.int32)
    max_utility = rng.integers(min_utility + 1, 100, None, np.int32)
    col_matrix = rng.integers(min_utility, max_utility, (num_rows, num_cols), np.int32)

    return row_matrix, col_matrix


def create_zero_sum_game(rng: np.random.Generator) -> np.ndarray:
    """Generate a random two-player zero-sum game."""

    return create_general_sum_game(rng)[0]


def create_game_with_dominated_strategies(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a game with a guaranteed sequence of strictly dominated strategies."""

    num_rows = rng.integers(3, 20, None, np.int32)
    num_cols = rng.integers(3, 20, None, np.int32)
    min_utility = rng.integers(-100, 0, None, np.int32)
    max_utility = rng.integers(min_utility + 1, 100, None, np.int32)

    # Determine the size of the unsolvable core matrix
    core_rows = rng.integers(1, num_rows, None, np.int32)
    core_cols = rng.integers(1, num_cols, None, np.int32)

    # Determine the random sequence of eliminations
    elimination_order = ['row'] * (num_rows - core_rows) + ['col'] * (num_cols - core_cols)
    rng.shuffle(elimination_order)

    # Generate the random "core" of the game, which is the final solution
    row_matrix = rng.integers(min_utility, max_utility + 1, (core_rows, core_cols), np.int32)
    col_matrix = rng.integers(min_utility, max_utility + 1, (core_rows, core_cols), np.int32)

    # Iteratively add the dominated strategies
    for player in elimination_order:
        current_rows, current_cols = row_matrix.shape

        if player == 'row':
            new_row_p1 = rng.integers(min_utility, max_utility + 1, (1, current_cols), np.int32)
            new_row_p2 = rng.integers(min_utility, max_utility + 1, (1, current_cols), np.int32)
            row_matrix = np.vstack([row_matrix, new_row_p1])
            col_matrix = np.vstack([col_matrix, new_row_p2])

            dominated_idx, dominating_idx = current_rows, rng.choice(current_rows)
            penalty_vector = rng.integers(1, 6, current_cols, np.int32)
            row_matrix[dominated_idx, :] = row_matrix[dominating_idx, :] - penalty_vector

        else:
            new_col_p1 = rng.integers(min_utility, max_utility + 1, (current_rows, 1), np.int32)
            new_col_p2 = rng.integers(min_utility, max_utility + 1, (current_rows, 1), np.int32)
            row_matrix = np.hstack([row_matrix, new_col_p1])
            col_matrix = np.hstack([col_matrix, new_col_p2])

            dominated_idx, dominating_idx = current_cols, rng.choice(current_cols)
            penalty_vector = rng.integers(1, 6, current_rows, np.int32)
            col_matrix[:, dominated_idx] = col_matrix[:, dominating_idx] - penalty_vector

    return row_matrix, col_matrix


def create_pure_strategy(num_actions: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random pure strategy."""

    strategy = np.zeros(num_actions, np.float32)
    index = rng.integers(0, num_actions, None, np.int32)
    strategy[index] = 1.0

    return strategy


def create_mixed_strategy(num_actions: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random mixed strategy."""

    strategy = rng.random(num_actions).astype(np.float32)
    strategy = strategy / np.sum(strategy)

    return strategy


def parameterize_general_sum_tests(rng: np.random.Generator) -> Generator:
    while True:
        row_matrix, col_matrix = create_general_sum_game(rng)

        if rng.random() < 0.5:
            # 50% of the time, use a pure strategy
            row_strategy = create_pure_strategy(row_matrix.shape[0], rng)
            col_strategy = create_pure_strategy(col_matrix.shape[1], rng)
        else:
            # 50% of the time, use a mixed strategy
            row_strategy = create_mixed_strategy(row_matrix.shape[0], rng)
            col_strategy = create_mixed_strategy(col_matrix.shape[1], rng)

        yield row_matrix, col_matrix, row_strategy, col_strategy


def parameterize_zero_sum_tests(rng: np.random.Generator) -> Generator:
    while True:
        row_matrix = create_zero_sum_game(rng)

        if rng.random() < 0.5:
            # 50% of the time, use a pure strategy
            row_strategy = create_pure_strategy(row_matrix.shape[0], rng)
            col_strategy = create_pure_strategy(row_matrix.shape[1], rng)
        else:
            # 50% of the time, use a mixed strategy
            row_strategy = create_mixed_strategy(row_matrix.shape[0], rng)
            col_strategy = create_mixed_strategy(row_matrix.shape[1], rng)

        yield row_matrix, -row_matrix, row_strategy, col_strategy


def parameterize_dominance_tests(rng: np.random.Generator) -> Generator:
    while True:
        yield create_game_with_dominated_strategies(rng)
