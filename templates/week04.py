#!/usr/bin/env python3

import numpy as np


def find_nash_equilibrium(row_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find a Nash equilibrium in a zero-sum normal-form game using linear programming.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A strategy profile that forms a Nash equilibrium
    """

    raise NotImplementedError


def find_correlated_equilibrium(row_matrix: np.ndarray, col_matrix: np.ndarray) -> np.ndarray:
    """Find a correlated equilibrium in a normal-form game using linear programming.

    While the cost vector could be selected to optimize a particular objective, such as
    maximizing the sum of players’ utilities, the reference solution sets it to the zero
    vector to ensure reproducibility during testing.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix

    Returns
    -------
    np.ndarray
        A distribution over joint actions that forms a correlated equilibrium
    """

    raise NotImplementedError


def main() -> None:
    pass


if __name__ == '__main__':
    main()
