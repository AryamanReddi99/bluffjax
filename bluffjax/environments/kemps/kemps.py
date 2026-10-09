"""
Kemps environment.

A partnership card game with simultaneous moves; one hand is one episode. The
rules follow pagat.com (https://www.pagat.com/commerce/kemps.html) and
Wikipedia ("Kemps (card game)"), with the multi-team STOP KEMPS rule from
gamerules.com ("Kemps"):

- An even number n >= 4 of players form k = n / 2 teams of two. Partners sit
  opposite each other: seat i's partner is seat i + k (mod n), so team t is
  {t, t + k} and every player sits between two opponents.
- Each player is dealt four cards, and four cards are dealt face up to the
  centre. The rest form the stock.
- There are no turns. Every step all players act at once; each action is a
  game action plus a public signal in 0..comm_dim-1, which every player sees
  at the next step (the partner signal; it has no fixed meaning).
- Game actions:
    swap (lose rank, gain rank): put a card of the lose rank from the hand into
        the centre and take a centre card of the gain rank;
    NOOP;
    KEMPS: claims that the caller's partner holds four of a kind;
    STOP KEMPS d (d = 1..k-1): claims that a player of the team of the player
        d seats after the caller (seats i + d and i + d + k) holds four of a
        kind. With two teams this is the single opposing team.
- Swaps are applied one at a time in the step's player order. A swap that is
  no longer possible when its turn comes (another player took the card) does
  nothing.
- If every player plays NOOP, the centre is swept away and four new cards are
  dealt from the stock. If the stock can't refill the centre, the hand ends
  with no score ("Real Deal").
- A declaration ends the hand at once. It is judged on the hands at the start
  of the step (this step's swaps don't happen):
    KEMPS right: every other team gets a letter; wrong: the caller's team does.
    STOP KEMPS right: the accused team gets a letter; wrong: the caller's team
    does.
- Simultaneous declarations: only one counts, the first declarer in the step's
  player order (a uniformly random permutation drawn each step, the same one
  that orders the swaps). KEMPS and STOP KEMPS have equal standing: the first
  call wins the race, whatever its type.
- Rewards (terminal, zero-sum, equal for both partners): with letters
  l_t in {0, 1} for each team t, team t gets k / (k - 1) * (mean(l) - l_t).
  With two teams this is +1 for the winners and -1 for the losers. With more
  teams, a team that takes the only letter gets -1 and every other team
  +1 / (k - 1); a right KEMPS gives the caller's team +1 and every other team
  -1 / (k - 1).
- Hands that reach `horizon` steps are cut off with zero reward.

Seat indices are an internal ordering only: the deal and the step's player
order are uniformly random, and the observation is relative to the observer.
"""

import jax
import jax.numpy as jnp
from jax import lax
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


def _cards_to_binary(cards: IntArray, deck_size: int) -> FloatArray:
    """Binary vector of shape (deck_size,) marking the given card indices."""
    return jnp.zeros(deck_size, dtype=jnp.float32).at[cards].set(1.0)


@struct.dataclass
class KempsState:
    """Kemps game state. Card c has rank c // num_suits."""

    agent_hands: IntArray  # (num_agents, hand_size) card indices
    agent_hand_counts: IntArray  # (num_agents, num_ranks) cards of each rank in hand
    center_cards: IntArray  # (4,) face-up centre cards
    center_counts: IntArray  # (num_ranks,) centre cards of each rank
    deck: IntArray  # stock, dealt from deck_idx onwards
    deck_idx: IntArray
    communication: FloatArray  # (num_agents, comm_dim) one-hot signals of the last step
    absorbing: BoolArray
    done: bool
    timestep: int


class Kemps(ParallelEnv):
    """Kemps parallel environment: n / 2 teams of two, simultaneous moves."""

    def __init__(
        self,
        num_agents: int = 4,
        num_ranks: int = 13,
        hand_size: int = 4,
        num_suits: int = 4,
        comm_dim: int = 2,
        horizon: int = 200,
    ) -> None:
        super().__init__(num_agents=num_agents, horizon=horizon)
        if num_agents < 4 or num_agents % 2 != 0:
            raise ValueError(f"Kemps needs an even number of at least 4 players, got {num_agents}")
        if hand_size < 4 or num_suits < 4:
            raise ValueError("Four of a kind needs hand_size >= 4 and num_suits >= 4")
        if comm_dim < 1:
            raise ValueError("comm_dim must be at least 1")
        self.num_teams = num_agents // 2
        self.num_ranks = num_ranks
        self.hand_size = hand_size
        self.num_suits = num_suits
        self.comm_dim = comm_dim
        self.deck_size = num_ranks * num_suits
        self.num_center = 4
        min_deck = num_agents * hand_size + self.num_center
        if self.deck_size < min_deck:
            raise ValueError(
                f"Deck too small: need {min_deck} cards, have {self.deck_size} "
                f"(num_ranks={num_ranks} * num_suits={num_suits})"
            )
        # Game actions: num_ranks**2 swaps, NOOP, KEMPS, then one STOP KEMPS per
        # opposing team. Each is paired with every signal.
        self.action_noop = num_ranks * num_ranks
        self.action_kemps = self.action_noop + 1
        self.num_game_actions = self.action_kemps + self.num_teams
        self.num_actions = self.num_game_actions * comm_dim
        self.obs_dim = 2 * self.deck_size + num_agents * comm_dim

    def _team_id(self, agent_idx: IntArray) -> IntArray:
        """Team t is the pair of seats {t, t + n/2}."""
        return agent_idx % self.num_teams

    def _partner_idx(self, agent_idx: IntArray) -> IntArray:
        """Partners sit opposite each other."""
        return (agent_idx + self.num_teams) % self.num_agents

    def _rank_counts(self, cards: IntArray) -> IntArray:
        """Number of cards of each rank, shape (num_ranks,)."""
        return jnp.bincount(cards // self.num_suits, length=self.num_ranks)

    @partial(jax.jit, static_argnums=(0,))
    def obs_from_state(self, state: KempsState) -> FloatArray:
        """
        Observation per agent: (num_agents, obs_dim), everything relative to the
        observer.
        - Own hand: (deck_size,) binary
        - Centre cards: (deck_size,) binary
        - Signals of the last step: (num_agents, comm_dim) one-hot, row j is the
          player j seats after the observer (row 0 is the observer, row n/2 the
          partner); all zeros before the first step
        """
        own_hand = jax.vmap(lambda h: _cards_to_binary(h, self.deck_size))(
            state.agent_hands
        )
        center = _cards_to_binary(state.center_cards, self.deck_size)
        center = jnp.broadcast_to(center, (self.num_agents, self.deck_size))
        signals = self._rel_array(state.communication).reshape(self.num_agents, -1)
        return jnp.concatenate([own_hand, center, signals], axis=1)

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: KempsState) -> BoolArray:
        """(num_agents, num_actions): each game action is legal with any signal.

        A swap needs the lose rank in hand and the gain rank in the centre.
        NOOP and the declarations are always legal: in the real game anyone may
        call at any time, and masking a call on the hidden hands would leak them.
        """
        has_lose = state.agent_hand_counts[:, :, None] >= 1
        center_has_gain = state.center_counts[None, None, :] >= 1
        swap_valid = (has_lose & center_has_gain).reshape(self.num_agents, -1)
        other_valid = jnp.ones((self.num_agents, 1 + self.num_teams), dtype=bool)
        game_avail = jnp.concatenate([swap_valid, other_valid], axis=1)
        return jnp.repeat(game_avail, self.comm_dim, axis=1)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: PRNGKeyArray) -> tuple[KempsState, FloatArray]:
        """Shuffle, deal hand_size cards to each player and 4 to the centre."""
        deck = jax.random.permutation(rng, self.deck_size)
        num_dealt = self.hand_size * self.num_agents
        agent_hands = deck[:num_dealt].reshape(self.num_agents, self.hand_size)
        center_cards = deck[num_dealt : num_dealt + self.num_center]
        state = KempsState(
            agent_hands=agent_hands,
            agent_hand_counts=jax.vmap(self._rank_counts)(agent_hands),
            center_cards=center_cards,
            center_counts=self._rank_counts(center_cards),
            deck=deck[num_dealt + self.num_center :],
            deck_idx=jnp.int32(0),
            communication=jnp.zeros((self.num_agents, self.comm_dim), dtype=jnp.float32),
            absorbing=jnp.zeros(self.num_agents, dtype=bool),
            done=False,
            timestep=0,
        )
        return state, self.obs_from_state(state)

    def _apply_swaps(
        self, state: KempsState, game_action: IntArray, order: IntArray
    ) -> KempsState:
        """Apply the swap actions one at a time, players in `order`."""
        num_swaps = self.num_ranks * self.num_ranks

        def swap_one(carry, seat):
            hands, hand_counts, center, center_counts = carry
            action = game_action[seat]
            lose_rank = jnp.minimum(action // self.num_ranks, self.num_ranks - 1)
            gain_rank = action % self.num_ranks
            valid = (
                (action < num_swaps)
                & (hand_counts[seat, lose_rank] >= 1)
                & (center_counts[gain_rank] >= 1)
            )
            hand_pos = jnp.argmax(hands[seat] // self.num_suits == lose_rank)
            center_pos = jnp.argmax(center // self.num_suits == gain_rank)
            give = hands[seat, hand_pos]
            take = center[center_pos]
            hands = hands.at[seat, hand_pos].set(jnp.where(valid, take, give))
            center = center.at[center_pos].set(jnp.where(valid, give, take))
            delta = valid * (
                jax.nn.one_hot(gain_rank, self.num_ranks, dtype=jnp.int32)
                - jax.nn.one_hot(lose_rank, self.num_ranks, dtype=jnp.int32)
            )
            hand_counts = hand_counts.at[seat].add(delta)
            center_counts = center_counts - delta
            return (hands, hand_counts, center, center_counts), None

        (hands, hand_counts, center, center_counts), _ = lax.scan(
            swap_one,
            (
                state.agent_hands,
                state.agent_hand_counts,
                state.center_cards,
                state.center_counts,
            ),
            order,
        )
        return state.replace(
            agent_hands=hands,
            agent_hand_counts=hand_counts,
            center_cards=center,
            center_counts=center_counts,
        )

    def _refresh_center(self, state: KempsState) -> KempsState:
        """Sweep the centre away and deal 4 new cards (the caller checks the stock)."""
        new_center = lax.dynamic_slice(state.deck, (state.deck_idx,), (self.num_center,))
        return state.replace(
            center_cards=new_center,
            center_counts=self._rank_counts(new_center),
            deck_idx=state.deck_idx + self.num_center,
        )

    def _declaration_rewards(
        self, state: KempsState, game_action: IntArray, order: IntArray
    ) -> tuple[BoolArray, FloatArray]:
        """Whether anyone declared, and the rewards of the first declaration in `order`."""
        k = self.num_teams
        is_declare = game_action >= self.action_kemps
        caller = order[jnp.argmax(is_declare[order])]
        call = game_action[caller]
        caller_team = self._team_id(caller)
        has_four = (state.agent_hand_counts >= 4).any(axis=1)
        team_has_four = has_four[:k] | has_four[k:]

        kemps_right = has_four[self._partner_idx(caller)]
        accused_team = self._team_id(caller + call - self.action_kemps)
        stop_right = team_has_four[accused_team]
        caller_letter = jax.nn.one_hot(caller_team, k)
        letters = jnp.where(
            call == self.action_kemps,
            jnp.where(kemps_right, 1.0 - caller_letter, caller_letter),
            jnp.where(stop_right, jax.nn.one_hot(accused_team, k), caller_letter),
        )
        team_rewards = k / (k - 1) * (letters.mean() - letters)
        rewards = team_rewards[self._team_id(self.agent_idxs)].astype(jnp.float32)
        return is_declare.any(), rewards

    def _step_in_order(
        self, state: KempsState, action: IntArray, order: IntArray
    ) -> tuple[KempsState, FloatArray, FloatArray, BoolArray, bool, dict[str, Any]]:
        """step_env with a given player order (first to act first)."""
        game_action = action // self.comm_dim
        signal = action % self.comm_dim

        any_declare, declare_rewards = self._declaration_rewards(state, game_action, order)
        all_noop = (game_action == self.action_noop).all()
        can_refill = state.deck_idx + self.num_center <= state.deck.shape[0]

        next_state = lax.cond(
            any_declare,
            lambda: state,
            lambda: lax.cond(
                all_noop & can_refill,
                lambda: self._refresh_center(state),
                lambda: self._apply_swaps(state, game_action, order),
            ),
        )
        real_deal = ~any_declare & all_noop & ~can_refill
        done = any_declare | real_deal | (state.timestep + 1 >= self.horizon)
        rewards = jnp.where(any_declare, declare_rewards, 0.0)
        absorbing = jnp.broadcast_to(done, (self.num_agents,))
        next_state = next_state.replace(
            communication=jax.nn.one_hot(signal, self.comm_dim, dtype=jnp.float32),
            absorbing=absorbing,
            done=done,
            timestep=state.timestep + 1,
        )
        obs = self.obs_from_state(next_state)
        return next_state, obs, rewards, absorbing, done, {}

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        rng: PRNGKeyArray,
        state: KempsState,
        action: IntArray,
    ) -> tuple[
        KempsState,
        FloatArray,
        FloatArray,
        BoolArray,
        bool,
        dict[str, Any],
    ]:
        """One step. action: (num_agents,), game_action * comm_dim + signal.

        The player order for this step (who swaps first, whose declaration
        counts) is a uniformly random permutation of the seats.
        """
        order = jax.random.permutation(rng, self.num_agents)
        return self._step_in_order(state, action, order)

    def observation_space(self) -> Discrete:
        return Discrete(self.obs_dim)

    def action_space(self) -> Discrete:
        return Discrete(self.num_actions)
