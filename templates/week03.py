#!/usr/bin/env python3

import numpy as np


def compute_deltas(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    row_strategy: np.ndarray,
    col_strategy: np.ndarray,
) -> np.ndarray:
    """Compute players' incentives to deviate from their strategies.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    np.ndarray
        Each player's incentive to deviate
    """

    raise NotImplementedError


def compute_nash_conv(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    row_strategy: np.ndarray,
    col_strategy: np.ndarray,
) -> float:
    """Compute the NashConv value of a given strategy profile.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    float
        The NashConv value of the given strategy profile
    """

    raise NotImplementedError


def compute_exploitability(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    row_strategy: np.ndarray,
    col_strategy: np.ndarray,
) -> float:
    """Compute the exploitability of a given strategy profile.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    row_strategy : np.ndarray
        The row player's strategy
    col_strategy : np.ndarray
        The column player's strategy

    Returns
    -------
    float
        The exploitability value of the given strategy profile
    """

    raise NotImplementedError


def fictitious_play(
    row_matrix: np.ndarray, col_matrix: np.ndarray, num_iters: int, naive: bool
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run Fictitious Play for a given number of epochs.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    num_iters : int
        The number of iterations to run the algorithm for
    naive : bool
        Whether to calculate the best response against the last
        opponent's strategy or the average opponent's strategy

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        The sequence of average strategy profiles produced by the algorithm
    """

    raise NotImplementedError


def plot_exploitability(
    row_matrix: np.ndarray,
    col_matrix: np.ndarray,
    strategies: list[tuple[np.ndarray, np.ndarray]],
    label: str,
) -> list[float]:
    """Compute and plot the exploitability of a sequence of strategy profiles.

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    col_matrix : np.ndarray
        The column player's payoff matrix
    strategies : list[tuple[np.ndarray, np.ndarray]]
        The sequence of strategy profiles
    label : str
        The name of the algorithm that produced `strategies`

    Returns
    -------
    list[float]
        A sequence of exploitability values, one for each strategy profile
    """

    raise NotImplementedError


def main() -> None:
    pass


if __name__ == '__main__':
    main()
