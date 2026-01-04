from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pgx.leduc_holdem
from pgx._src.struct import dataclass


@dataclass
class State(pgx.leduc_holdem.State):
    is_chance_node: jax.Array = jnp.array(False, jnp.bool_)
    chance_strategy: jax.Array = jnp.full((3,), -1, jnp.float32)


class LeducPoker(pgx.leduc_holdem.LeducHoldem):
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
        remaining_cards = jnp.full((4,), 2, jnp.int32).at[dealt_cards].add(-1)[:3]

        is_chance_node = jnp.any(dealt_cards[:2] == -1)
        chance_strategy = remaining_cards / jnp.sum(remaining_cards)

        current_player = jnp.where(is_chance_node, -1, 0)
        action_mask = jnp.where(
            is_chance_node,
            state.legal_action_mask,
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
        remaining_cards = jnp.full((4,), 2, jnp.int32).at[next_state._cards].add(-1)[:3]

        is_chance_node = (
            (next_state._round > 0) & (next_state._cards[-1] == -1) & ~next_state.terminated
        )
        chance_strategy = remaining_cards / jnp.sum(remaining_cards)

        current_player = jnp.where(is_chance_node, -1, next_state.current_player)
        action_mask = jnp.where(
            is_chance_node,
            remaining_cards > 0,
            next_state.legal_action_mask,
        )

        return replace(
            next_state,
            current_player=current_player,
            legal_action_mask=action_mask,
            is_chance_node=is_chance_node,
            chance_strategy=chance_strategy,
        )


class LeducPokerNumpy:
    def __init__(self) -> None:
        self.env = LeducPoker()
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
