#!/usr/bin/env python3

import numpy as np


def double_oracle(
    row_matrix: np.ndarray, eps: float, rng: np.random.Generator
) -> tuple[list[np.ndarray, np.ndarray], list[np.ndarray, np.ndarray]]:
    """Run Double Oracle until a termination condition is met.

    The reference implementation generates the initial restricted game by
    randomly sampling one pure action for each player using `rng.integers`.

    The algorithm terminates when either:
        1. the difference between the upper and the lower bound on the game value drops below `eps`
        2. both players' best responses are already contained in the current restricted game

    Parameters
    ----------
    row_matrix : np.ndarray
        The row player's payoff matrix
    eps : float
        The required accuracy for the approximate Nash equilibrium
    rng : np.random.Generator
        A random number generator

    Returns
    -------
    tuple[list[np.ndarray, np.ndarray], list[np.ndarray, np.ndarray]]
        A tuple containing a sequence of strategy profiles and a sequence of corresponding supports
    """

    raise NotImplementedError


def main() -> None:
    pass


if __name__ == '__main__':
    main()
