#!/usr/bin/env python3

import numpy as np

from kuhn_poker import KuhnPokerNumpy as KuhnPoker
from week01 import evaluate_zero_sum
from week04 import find_nash_equilibrium
from week07 import evaluate, traverse_tree


def convert_to_normal_form(*args, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Convert an extensive-form game into an equivalent normal-form representation.

    Feel free to conceptually split this function into smaller functions that compute
        - the set of pure strategies for a player, e.g. `_collect_pure_strategies`
        - the expected utility of a pure strategy profile, e.g. `_compute_expected_utility`

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A pair of payoff matrices for the two players in the resulting normal-form game.
    """

    raise NotImplementedError


def convert_to_sequence_form(*args, **kwargs) -> tuple[np.ndarray, ...]:
    """Convert an extensive-form game into its sequence-form representation.

    The sequence-form representation consists of:
        - The sequence-form payoff matrices for both players
        - The realization-plan constraint matrices and vectors for both players

    Feel free to conceptually split this function into smaller functions that compute
        - the sequences for a player, e.g. `_collect_sequences`
        - the sequence-form payoff matrix of a player, e.g. `_compute_sequence_form_payoff_matrix`
        - the realization-plan constraints of a player, e.g. `_compute_realization_plan_constraints`

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the sequence-form payoff matrices and realization-plan constraints.
    """

    raise NotImplementedError


def find_nash_equilibrium_sequence_form(*args, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Find a Nash equilibrium in a zero-sum extensive-form game using Sequence-form LP.

    This function is expected to received an extensive-form game as input
    and convert it to its sequence-form using `convert_to_sequence_form`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A pair of realization plans for the two players for representing a Nash equilibrium.
    """

    raise NotImplementedError


def convert_realization_plan_to_behavioural_strategy(*args, **kwargs):
    """Convert a realization plan to a behavioural strategy."""

    raise NotImplementedError


def main() -> None:
    # Replace each `...` with the arguments required by your implementation
    env = KuhnPoker()
    tree = traverse_tree(...)

    # Normal form
    row_matrix, col_matrix = convert_to_normal_form(...)
    print(f'Normal-form payoff matrices: {row_matrix.shape}, {col_matrix.shape}')

    row_strategy, col_strategy = find_nash_equilibrium(row_matrix)
    row_utility, col_utility = evaluate_zero_sum(row_matrix, row_strategy, col_strategy)
    print(f'Normal-form LP utilities: {row_utility:.5f}, {col_utility:.5f}')

    # Sequence form
    row_matrix, col_matrix, *constraints = convert_to_sequence_form(...)
    print(f'Sequence-form payoff matrices: {row_matrix.shape}, {col_matrix.shape}')
    print('Realization-plan constraints:', ', '.join(str(array.shape) for array in constraints))

    row_plan, col_plan = find_nash_equilibrium_sequence_form(...)
    row_utility, col_utility = row_plan @ row_matrix @ col_plan, row_plan @ col_matrix @ col_plan
    print(f'Sequence-form LP utilities: {row_utility:.5f}, {col_utility:.5f}')

    row_strategy = convert_realization_plan_to_behavioural_strategy(...)
    col_strategy = convert_realization_plan_to_behavioural_strategy(...)
    row_utility, col_utility = evaluate(...)
    print(f'Behavioural strategy utilities: {row_utility:.5f}, {col_utility:.5f}')


if __name__ == '__main__':
    main()
