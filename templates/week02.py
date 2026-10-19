#!/usr/bin/env python3

import numpy as np


def plot_best_response_value_function(row_matrix: np.ndarray, step_size: float) -> None:
    """Plot the best response value function for the row player in a 2xN zero-sum game.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    step_size : float
        The step size for the probability of the first action of the row player
    """

    raise NotImplementedError


def verify_support(
    matrix: np.ndarray, row_support: np.ndarray, col_support: np.ndarray
) -> np.ndarray | None:
    """Construct a system of linear equations to check whether there
    exists a candidate for a Nash equilibrium for the given supports.

    The returned strategy of the opponent (the column player of `matrix`) makes
    all actions in `row_support` equally good. This function is not supposed to
    check if actions outside of the given support lead to a higher payoff.

    To compute the strategy of the row player, call this function with the transposed
    payoff matrix of the column player and with the two supports swapped.

    The reference implementation uses `scipy.optimize.linprog`
    with the default solver -- 'highs'. You can find more information at
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html

    Parameters
    ----------
    matrix : np.ndarray
        A payoff matrix of the player whose actions correspond to rows
    row_support : np.ndarray
        The support of the player whose actions correspond to rows of `matrix`
    col_support : np.ndarray
        The support of the opponent whose actions correspond to columns of `matrix`

    Returns
    -------
    np.ndarray | None
        Probabilities of the opponent's actions in `col_support`, if they exist, otherwise `None`
    """

    raise NotImplementedError


def support_enumeration(
    row_matrix: np.ndarray, col_matrix: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run the Support Enumeration algorithm and return a list of all Nash equilibria

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        A list of strategy profiles corresponding to found Nash equilibria
    """

    raise NotImplementedError


def main() -> None:
    pass


if __name__ == '__main__':
    main()
