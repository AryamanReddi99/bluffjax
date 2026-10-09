"""
Goofspiel (Game of Pure Strategy) environment.

A simultaneous-move bidding card game. Each player holds the cards 1-13, and a
shuffled prize deck of the cards 1-13 is revealed one card per round. Every
round all players bid one card from their hand at the same time; the unique
highest bid wins the prize, worth its face value, and a tie for the highest
bid discards the prize. Bid cards are used up. The game ends after 13 rounds.

Rewards are zero-sum: each round the winner gets the prize value and every
other player loses value / (n - 1) (with two players, -value). A player's
return is therefore its points minus the average points of its opponents
(with two players, the final point difference).

Bids are assumed to come from get_avail_actions. A bid of a card that is no
longer in hand is replaced by the lowest card still in hand, which is played
instead, so a card can never be bid twice.
"""

import jax
import jax.numpy as jnp
from flax import struct
from functools import partial
from bluffjax.utils.typing import (
    Any,
    FloatArray,
    IntArray,
    BoolArray,
    PRNGKeyArray,
)
from bluffjax.environments.env import ParallelEnv
from bluffjax.environments.spaces import Discrete


@struct.dataclass
class GoofspielState:
    """Goofspiel game state. Card or prize c (0..12) has value c + 1."""

    player_hands: BoolArray  # (num_agents, 13) True while the card is in hand
    deck: IntArray  # (13,) prizes in order of revelation
    current_round: IntArray  # number of prizes already contested, 0..13
    points: FloatArray  # (num_agents,) cumulative prize points
    winners: BoolArray  # (num_agents,) at the end, True for the unique top scorer
    absorbing: BoolArray  # (num_agents,)
    done: bool
    timestep: int


class Goofspiel(ParallelEnv):
    """Goofspiel parallel environment."""

    def __init__(self, num_agents: int = 2, horizon: int = 14) -> None:
        super().__init__(num_agents=num_agents, horizon=horizon)
        if num_agents < 2:
            raise ValueError(f"Goofspiel needs at least 2 players, got {num_agents}")
        self.deck_size = 13
        self.total_points = self.deck_size * (self.deck_size + 1) / 2
        self.obs_dim = self.deck_size * (num_agents + 2) + num_agents
        self.num_actions = self.deck_size

    @partial(jax.jit, static_argnums=(0,))
    def obs_from_state(self, state: GoofspielState) -> FloatArray:
        """
        Observation per agent: (num_agents, obs_dim), relative to the observer.
        - Current prize card: one-hot (13), all zeros once the game is over
        - Prize cards already contested: binary (13)
        - Cards already bid: binary (num_agents, 13), row j is the player j
          seats after the observer (row 0 is the observer)
        - Points: (num_agents,), same order, divided by the 91 points in the deck
        """
        in_play = state.current_round < self.deck_size
        prize = jax.nn.one_hot(
            state.deck[jnp.minimum(state.current_round, self.deck_size - 1)],
            self.deck_size,
            dtype=jnp.float32,
        ) * in_play
        contested = jnp.zeros(self.deck_size, dtype=jnp.float32).at[state.deck].set(
            (jnp.arange(self.deck_size) < state.current_round).astype(jnp.float32)
        )
        public = jnp.broadcast_to(
            jnp.concatenate([prize, contested]), (self.num_agents, 2 * self.deck_size)
        )
        cards_bid = self._rel_array(~state.player_hands).reshape(self.num_agents, -1)
        points = self._rel_array(state.points / self.total_points)
        return jnp.concatenate([public, cards_bid, points], axis=1)

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: GoofspielState) -> BoolArray:
        """Available actions: (num_agents, 13), True if the card is in hand."""
        return state.player_hands & ~state.done

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: PRNGKeyArray) -> tuple[GoofspielState, FloatArray]:
        """Reset the environment."""
        state = GoofspielState(
            player_hands=jnp.ones((self.num_agents, self.deck_size), dtype=bool),
            deck=jax.random.permutation(rng, self.deck_size),
            current_round=jnp.int32(0),
            points=jnp.zeros(self.num_agents, dtype=jnp.float32),
            winners=jnp.zeros(self.num_agents, dtype=bool),
            absorbing=jnp.zeros(self.num_agents, dtype=bool),
            done=False,
            timestep=0,
        )
        return state, self.obs_from_state(state)

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        rng: PRNGKeyArray,
        state: GoofspielState,
        action: IntArray,
    ) -> tuple[
        GoofspielState,
        FloatArray,
        FloatArray,
        BoolArray,
        bool,
        dict[str, Any],
    ]:
        """Play one round: resolve the simultaneous bids for the current prize."""
        in_range = (action >= 0) & (action < self.deck_size)
        in_hand = in_range & state.player_hands[self.agent_idxs, action]
        bid = jnp.where(in_hand, action, jnp.argmax(state.player_hands, axis=1))

        prize_value = (state.deck[state.current_round] + 1).astype(jnp.float32)
        is_max = bid == jnp.max(bid)
        won = is_max & (is_max.sum() == 1)
        gained = jnp.where(won, prize_value, 0.0)
        rewards = gained - (gained.sum() - gained) / (self.num_agents - 1)

        new_points = state.points + gained
        new_hands = state.player_hands.at[self.agent_idxs, bid].set(False)
        new_round = state.current_round + 1
        game_done = new_round >= self.deck_size
        is_top = new_points == jnp.max(new_points)
        new_winners = game_done & is_top & (is_top.sum() == 1)
        new_absorbing = jnp.broadcast_to(game_done, (self.num_agents,))

        next_state = GoofspielState(
            player_hands=new_hands,
            deck=state.deck,
            current_round=new_round,
            points=new_points,
            winners=new_winners,
            absorbing=new_absorbing,
            done=game_done,
            timestep=state.timestep + 1,
        )
        info = {"points": new_points, "game_winner": new_winners}
        return next_state, self.obs_from_state(next_state), rewards, new_absorbing, game_done, info

    def observation_space(self) -> Discrete:
        """Observation space: obs_dim floats per agent (54 with two players)."""
        return Discrete(self.obs_dim)

    def action_space(self) -> Discrete:
        """Action space: 13 discrete choices per agent (the card to bid)."""
        return Discrete(self.num_actions)
