#!/usr/bin/env python3

import matplotlib.pyplot as plt

from kuhn_poker import KuhnPokerNumpy as KuhnPoker
from leduc_poker import LeducPokerNumpy as LeducPoker
from week07 import compute_exploitability, traverse_tree
from week09 import cfr


def discounted_cfr(*args, **kwargs):
    """Run the Discounted CFR algorithm for a given number of iterations."""

    raise NotImplementedError


def monte_carlo_cfr(*args, **kwargs):
    """Run the outcome-sampling version of Monte Carlo CFR for a given number of iterations."""

    raise NotImplementedError


def main() -> None:
    # Replace each `...` with the arguments required by your implementation
    for game_name, env in (('Kuhn Poker', KuhnPoker()), ('Leduc Poker', LeducPoker())):
        tree = traverse_tree(...)
        results = {
            'CFR': cfr(..., num_iters=500),
            'Discounted CFR': discounted_cfr(..., num_iters=500),
            'Monte Carlo CFR': monte_carlo_cfr(..., num_iters=10_000),
        }

        plt.figure()
        for label, strategies in results.items():
            exploitabilities = compute_exploitability(...)
            plt.plot(range(1, len(exploitabilities) + 1), exploitabilities, label=label)

        plt.title(f'Exploitability on {game_name}')
        plt.xlabel('Iterations')
        plt.ylabel('Exploitability')
        plt.xscale('log')
        plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
