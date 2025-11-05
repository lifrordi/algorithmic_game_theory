#!/usr/bin/env python3

import numpy as np


def regret_matching(regrets: np.ndarray) -> np.ndarray:
    """Generate a strategy based on the given cumulative regrets.

    Parameters
    ----------
    regrets : np.ndarray
        The vector containing cumulative regret of each action

    Returns
    -------
    np.ndarray
        The generated strategy
    """

    raise NotImplementedError


def regret_minimization(
    row_matrix: np.ndarray, col_matrix: np.ndarray, num_iters: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run Regret Minimization for a given number of iterations.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    num_iters : int
        The number of iterations to run the algorithm for

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        The sequence of `num_iters` average strategy profiles produced by the algorithm
    """

    raise NotImplementedError


def main() -> None:
    pass


if __name__ == '__main__':
    main()
