#!/usr/bin/env python3

import matplotlib.pyplot as plt

from kuhn_poker import KuhnPokerNumpy as KuhnPoker
from leduc_poker import LeducPokerNumpy as LeducPoker
from week07 import compute_exploitability, fictitious_play, traverse_tree


def cfr(*args, **kwargs):
    """Run the CFR algorithm for a given number of iterations."""

    raise NotImplementedError


def cfr_plus(*args, **kwargs):
    """Run the CFR+ algorithm for a given number of iterations."""

    raise NotImplementedError


def main() -> None:
    # Replace each `...` with the arguments required by your implementation
    for game_name, env in (('Kuhn Poker', KuhnPoker()), ('Leduc Poker', LeducPoker())):
        tree = traverse_tree(...)
        results = {
            'Fictitious Play': fictitious_play(..., num_iters=500),
            'CFR': cfr(..., num_iters=500),
            'CFR+': cfr_plus(..., num_iters=500),
        }

        plt.figure()
        for label, strategies in results.items():
            exploitabilities = compute_exploitability(...)
            plt.plot(range(1, len(exploitabilities) + 1), exploitabilities, label=label)

        plt.title(f'Exploitability on {game_name}')
        plt.xlabel('Iterations')
        plt.ylabel('Exploitability')
        plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
