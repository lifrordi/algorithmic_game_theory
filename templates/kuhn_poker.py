from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pgx.kuhn_poker
from pgx._src.struct import dataclass


@dataclass
class State(pgx.kuhn_poker.State):
    is_chance_node: jax.Array = jnp.array(False, jnp.bool_)
    chance_strategy: jax.Array = jnp.full((3,), -1, jnp.float32)


class KuhnPoker(pgx.kuhn_poker.KuhnPoker):
    def _init(self, key: jax.Array) -> State:
        state = super()._init(key)
        state = State(**vars(state))

        current_player = jnp.array(-1, jnp.int32)
        action_mask = jnp.ones((3,), jnp.bool_)
        dealt_cards = jnp.full_like(state._cards, -1, jnp.int32)

        is_chance_node = jnp.array(True, jnp.bool_)
        chance_strategy = jnp.full((3,), 1 / 3, jnp.float32)

        return replace(
            state,
            current_player=current_player,
            legal_action_mask=action_mask,
            is_chance_node=is_chance_node,
            chance_strategy=chance_strategy,
            _cards=dealt_cards,
        )

    def _step(self, state: State, action: jax.Array, key: jax.Array) -> State:
        return jax.lax.cond(
            state.is_chance_node, self._step_chance, self._step_player, state, action, key
        )

    def _step_chance(self, state: State, action: jax.Array, key: jax.Array) -> State:
        dealt_cards = state._cards.at[jnp.sum(state._cards != -1)].set(action)

        is_chance_node = jnp.any(dealt_cards == -1)
        chance_strategy = state.chance_strategy.at[action].set(0.0)
        chance_strategy = jnp.where(
            is_chance_node,
            chance_strategy / jnp.sum(chance_strategy),
            jnp.full((3,), -1, jnp.float32),
        )

        current_player = jnp.where(is_chance_node, -1, 0)
        action_mask = jnp.where(
            is_chance_node,
            state.legal_action_mask.at[action].set(False),
            jnp.array([True, True, False], jnp.bool_),
        )

        return replace(
            state,
            current_player=current_player,
            legal_action_mask=action_mask,
            is_chance_node=is_chance_node,
            chance_strategy=chance_strategy,
            _cards=dealt_cards,
        )

    def _step_player(self, state: State, action: jax.Array, key: jax.Array) -> State:
        next_state = super()._step(state, action, key)

        # Keep the action mask sizes consistent
        action_mask = jnp.zeros((3,), jnp.bool_).at[:2].set(next_state.legal_action_mask)

        return replace(next_state, legal_action_mask=action_mask)


class KuhnPokerNumpy:
    def __init__(self) -> None:
        self.env = KuhnPoker()
        self.env.init = jax.jit(self.env.init)
        self.env.step = jax.jit(self.env.step)

    def init(self, seed: int) -> State:
        state = self.env.init(jax.random.key(seed))
        state = jax.tree.map(lambda x: np.asarray(x), state)

        return state

    def step(self, state: State, action: int) -> State:
        state = self.env.step(state, jnp.array(action, jnp.int32))
        state = jax.tree.map(lambda x: np.asarray(x), state)

        return state
