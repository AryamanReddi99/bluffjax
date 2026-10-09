"""
7-Card Stud poker environment (limit betting, 2 to 10 players).

Deal
- Cards are dealt from one shuffled deck, street by street. On 3rd street
  every player gets two face-down hole cards and a face-up door card; on 4th,
  5th and 6th street one face-up card and on 7th street one face-down card,
  each only to the players still in the hand.
- Running out of cards: when the deck can't give every player still in the
  hand a card and keep one card for each later street, the street is dealt as
  a single face-up community card that all of them use as their card of that
  street. With up to 7 players this never happens. With 8 it can only happen on
  7th street and only when the deck is short, which is the standard stud rule.
  With 9 or 10 players it can also happen on 6th street. The deal works for
  up to 16 players.

Order of play
- Every player antes; the lowest door card (suits break ties, card // 13)
  posts the bring-in, and the player after it acts first on 3rd street.
- On 4th to 7th street the best hand showing acts first: pairs, two pair,
  trips and quads, then high cards (a 7th-street community card makes it a
  five-card board, ranked as a poker hand). Ties go to the tied player who
  comes first clockwise after the bring-in. The cards decide the bring-in, so
  no seat index is favoured.

Betting: limit, bets of small_bet on 3rd and 4th street and big_bet after,
at most allowed_raise_num raises per street. Rewards are net chips divided by
big_bet (big bets).

Observation (for the player to act; seat j is the player j places clockwise
from it, seat 0 is itself):
- [0, 52): own face-down cards (multi-hot).
- [52, 52 + 260 n): face-up cards, 52 * (5 j + t) + card for seat j and street
  t = 0 (3rd street, the door card) to 4 (7th street, only a community card).
  Only cards dealt so far are shown; a community card shows for every player
  who got it. Folded players' face-up cards stay visible.
- next n - 1: seat j = 1, ..., n - 1 has folded.
- next 25: raise count of each street (5 streets x one-hot of 0-4).
- next n: own position clockwise from the bring-in (one-hot).
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
from bluffjax.environments.env import AECEnv
from bluffjax.environments.spaces import Discrete
from bluffjax.utils.game_utils.poker_utils import (
    _card_rank,
    _card_suit,
    _compare_hands,
    _get_bring_in_idx,
    _score_five_card_hand,
    _score_visible_upcards,
)

# Card slots of a player: 0, 1 hole cards and 2 door card (3rd street), then
# one slot per street: 3 (4th), 4 (5th), 5 (6th), 6 (7th).
SLOT_STREET = jnp.array([0, 0, 0, 1, 2, 3, 4], dtype=jnp.int32)
# Face up unless the street's card was a community card (always face up).
SLOT_FACE_UP = jnp.array([False, False, True, True, True, True, False])


@struct.dataclass
class SevenCardStudState:
    deck: IntArray  # (52,) shuffled deck, dealt from the front
    num_dealt: IntArray  # cards taken from the deck so far
    agent_cards: IntArray  # (num_agents, 7) card per slot, -1 = not dealt
    community: BoolArray  # (5,) the street was dealt as one community card
    chips_in: FloatArray  # (num_agents,) - per-player contribution this hand
    bring_in_idx: IntArray  # who posts bring-in (3rd street)
    current_player_idx: IntArray
    stage: IntArray  # 0=3rd, 1=4th, 2=5th, 3=6th, 4=7th street
    raise_nums: IntArray  # (5,) - raises per betting round
    folded: BoolArray
    not_raise_num: IntArray
    absorbing: BoolArray
    done: bool
    timestep: int


class SevenCardStud(AECEnv):
    def __init__(self, num_agents: int = 2, horizon: int = 100_000) -> None:
        super().__init__(num_agents=num_agents, horizon=horizon)
        self.deck_size = 52
        self.num_streets = 5
        if not 2 <= num_agents <= (self.deck_size - (self.num_streets - 1)) // 3:
            # 3rd street must leave one card for each later street.
            raise ValueError(f"7-Card Stud needs 2 to 16 players, got {num_agents}")
        self.small_bet = 1
        self.big_bet = 2
        self.ante = 0.5
        self.bring_in = 0.5
        self.raise_amount_small = self.small_bet
        self.raise_amount_big = self.big_bet
        self.allowed_raise_num = 4
        self.num_betting_rounds = self.num_streets
        # own face-down cards, face-up cards (n seats x 5 streets), opponents
        # folded, raise history, position after the bring-in
        self.obs_dim = (
            52
            + num_agents * self.num_streets * 52
            + (num_agents - 1)
            + self.num_betting_rounds * (self.allowed_raise_num + 1)
            + num_agents
        )

    def _next_active(self, idx: IntArray, folded: BoolArray) -> IntArray:
        """First player clockwise after idx who hasn't folded."""
        candidates = (idx + jnp.arange(1, self.num_agents + 1)) % self.num_agents
        return candidates[jnp.argmax(~folded[candidates])]

    def _after_bring_in(self, bring_in_idx: IntArray) -> IntArray:
        """All seats clockwise, starting after the bring-in."""
        return (bring_in_idx + 1 + jnp.arange(self.num_agents)) % self.num_agents

    def _deal_street(
        self, state: SevenCardStudState, street: IntArray, folded: BoolArray
    ) -> tuple[IntArray, IntArray, BoolArray]:
        """Deals street (1-4) to the players still in the hand.

        Returns the new agent_cards, num_dealt and community.
        """
        active = ~folded
        num_active = jnp.sum(active)
        later_streets = self.num_streets - 1 - street
        use_community = (
            self.deck_size - state.num_dealt - num_active < later_streets
        )
        # Individual cards go out clockwise from the bring-in's left.
        order = self._after_bring_in(state.bring_in_idx)
        rank = jnp.zeros(self.num_agents, dtype=jnp.int32).at[order].set(
            jnp.cumsum(active[order]) - 1
        )
        pos = state.num_dealt + jnp.where(use_community, 0, rank)
        cards = jnp.where(active, state.deck[jnp.minimum(pos, self.deck_size - 1)], -1)
        street = jnp.minimum(street, self.num_streets - 1)
        agent_cards = state.agent_cards.at[:, street + 2].set(cards)
        num_dealt = state.num_dealt + jnp.where(use_community, 1, num_active)
        community = state.community.at[street].set(use_community)
        return agent_cards, num_dealt, community

    def _first_to_act_by_upcards(
        self,
        agent_cards: IntArray,
        community: BoolArray,
        folded: BoolArray,
        street: IntArray,
        bring_in_idx: IntArray,
    ) -> IntArray:
        """Best hand showing on street 1-4; ties go to the first tied player
        clockwise after the bring-in."""
        up = jnp.maximum(agent_cards[:, 2:], 0)  # face-up card of streets 0-4
        num_up = jnp.where(
            (street == 4) & community[4], 5, jnp.minimum(street + 1, 4)
        )

        def score_board(k):
            return jax.vmap(_score_visible_upcards)(up[:, :k])

        scores = jnp.select(
            [num_up == 2, num_up == 3, num_up == 4],
            [score_board(2), score_board(3), score_board(4)],
            jax.vmap(lambda c: _score_five_card_hand(_card_rank(c), _card_suit(c)))(
                up
            ),
        )
        scores = jnp.where(folded, -1, scores)
        order = self._after_bring_in(bring_in_idx)
        return order[jnp.argmax(scores[order])]

    @partial(jax.jit, static_argnums=(0,))
    def obs_from_state(self, state: SevenCardStudState) -> FloatArray:
        """Observation of the player to act (layout in the module docstring)."""
        current = state.current_player_idx
        seats = (current + jnp.arange(self.num_agents)) % self.num_agents
        cards = state.agent_cards[seats]  # (n, 7), relative seat order
        face_up = SLOT_FACE_UP | state.community[SLOT_STREET]
        own_down = jnp.where(face_up, -1, cards[0])
        up = jnp.where(face_up[2:], cards[:, 2:], -1)  # (n, 5) by street
        position = (current - state.bring_in_idx) % self.num_agents
        return jnp.concatenate(
            [
                jax.nn.one_hot(own_down, 52).sum(axis=0),
                jax.nn.one_hot(up, 52).reshape(-1),
                state.folded[seats[1:]].astype(jnp.float32),
                jax.nn.one_hot(state.raise_nums, self.allowed_raise_num + 1).reshape(
                    -1
                ),
                jax.nn.one_hot(position, self.num_agents),
            ]
        ).astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: SevenCardStudState) -> BoolArray:
        """
        Actions: 0=call, 1=raise, 2=fold, 3=check
        Same logic as Texas Limit Hold'em.
        """
        current_player = state.current_player_idx
        player_chips = state.chips_in[current_player]
        max_chips = jnp.max(state.chips_in)

        avail_actions = jnp.ones(4, dtype=bool)
        can_raise = state.raise_nums[state.stage] < self.allowed_raise_num
        avail_actions = avail_actions.at[1].set(can_raise)
        can_call = player_chips < max_chips
        avail_actions = avail_actions.at[0].set(can_call)
        can_check = player_chips == max_chips
        avail_actions = avail_actions.at[3].set(can_check)
        return avail_actions

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: PRNGKeyArray) -> tuple[SevenCardStudState, FloatArray]:
        """Reset: shuffle, deal 3rd street, post antes and the bring-in."""
        rng_shuffle, _ = jax.random.split(rng)
        deck = jax.random.permutation(rng_shuffle, self.deck_size).astype(jnp.int32)
        num_dealt = 3 * self.num_agents
        agent_cards = (
            jnp.full((self.num_agents, 7), -1, dtype=jnp.int32)
            .at[:, :3]
            .set(deck[:num_dealt].reshape(self.num_agents, 3))
        )

        # Bring-in: lowest door card (slot 2)
        bring_in_idx = _get_bring_in_idx(agent_cards[:, 2])

        # Ante: each player posts ante; bring-in: bring_in_idx posts bring-in
        chips_in = jnp.full(self.num_agents, self.ante, dtype=jnp.float32)
        chips_in = chips_in.at[bring_in_idx].add(self.bring_in)

        # First to act on 3rd street: the player after the bring-in
        folded = jnp.zeros(self.num_agents, dtype=bool)
        current_player_idx = self._next_active(bring_in_idx, folded)

        state = SevenCardStudState(
            deck=deck,
            num_dealt=jnp.int32(num_dealt),
            agent_cards=agent_cards,
            community=jnp.zeros(self.num_streets, dtype=bool),
            chips_in=chips_in,
            bring_in_idx=bring_in_idx,
            current_player_idx=current_player_idx,
            stage=jnp.int32(0),
            raise_nums=jnp.zeros(self.num_betting_rounds, dtype=jnp.int32),
            folded=folded,
            not_raise_num=jnp.int32(0),
            absorbing=jnp.zeros(self.num_agents, dtype=bool),
            done=jnp.array(False),
            timestep=0,
        )

        obs = self.obs_from_state(state)
        return state, obs

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self, rng: PRNGKeyArray, state: SevenCardStudState, action: IntArray
    ) -> tuple[
        SevenCardStudState, FloatArray, FloatArray, BoolArray, bool, dict[str, Any]
    ]:
        """
        Execute one step. Actions: 0=call, 1=raise, 2=fold, 3=check.
        """
        current_player = state.current_player_idx
        max_chips = jnp.max(state.chips_in)
        player_chips = state.chips_in[current_player]

        # Raise amount: small bet for stage 0,1; big bet for stage 2,3,4
        raise_amount = jnp.where(
            state.stage >= 2, self.raise_amount_big, self.raise_amount_small
        )
        current_round_raises = state.raise_nums[state.stage]

        def process_call():
            diff = max_chips - player_chips
            new_chips_in = state.chips_in.at[current_player].add(diff)
            return new_chips_in, state.raise_nums, state.not_raise_num + 1, state.folded

        def process_raise():
            diff = max_chips - player_chips + raise_amount
            new_chips_in = state.chips_in.at[current_player].add(diff)
            new_raise_nums = state.raise_nums.at[state.stage].set(
                current_round_raises + 1
            )
            return new_chips_in, new_raise_nums, jnp.int32(1), state.folded

        def process_fold():
            new_folded = state.folded.at[current_player].set(True)
            return state.chips_in, state.raise_nums, state.not_raise_num, new_folded

        def process_check():
            return (
                state.chips_in,
                state.raise_nums,
                state.not_raise_num + 1,
                state.folded,
            )

        new_chips_in, new_raise_nums, new_not_raise_num, new_folded = lax.switch(
            action, [process_call, process_raise, process_fold, process_check]
        )

        num_active_players = jnp.sum(~new_folded)
        round_over = new_not_raise_num >= num_active_players

        new_stage = jnp.where(round_over, state.stage + 1, state.stage)
        new_not_raise_num = jnp.where(round_over, jnp.int32(0), new_not_raise_num)
        game_done = (num_active_players <= 1) | (new_stage >= self.num_streets)

        # Next street: deal it, then the best hand showing acts first.
        deal = round_over & ~game_done
        dealt_cards, dealt_num, dealt_community = self._deal_street(
            state, new_stage, new_folded
        )
        agent_cards = jnp.where(deal, dealt_cards, state.agent_cards)
        num_dealt = jnp.where(deal, dealt_num, state.num_dealt)
        community = jnp.where(deal, dealt_community, state.community)
        first_to_act = self._first_to_act_by_upcards(
            agent_cards, community, new_folded, new_stage, state.bring_in_idx
        )
        new_current_player_idx = jnp.where(
            round_over, first_to_act, self._next_active(current_player, new_folded)
        )

        next_state = SevenCardStudState(
            deck=state.deck,
            num_dealt=num_dealt,
            agent_cards=agent_cards,
            community=community,
            chips_in=new_chips_in,
            bring_in_idx=state.bring_in_idx,
            current_player_idx=new_current_player_idx,
            stage=new_stage,
            raise_nums=new_raise_nums,
            folded=new_folded,
            not_raise_num=new_not_raise_num,
            absorbing=jnp.broadcast_to(game_done, (self.num_agents,)),
            done=game_done,
            timestep=state.timestep + 1,
        )

        obs = self.obs_from_state(next_state)

        def compute_rewards() -> FloatArray:
            # At a showdown every player still in has all 7 cards.
            hands = jnp.maximum(agent_cards, 0)
            winners_by_score = _compare_hands(hands, new_folded)
            winners = jnp.where(num_active_players == 1, ~new_folded, winners_by_score)
            pot_total = jnp.sum(new_chips_in).astype(jnp.float32)
            num_winners = jnp.sum(winners).astype(jnp.float32)
            chips_gained = jnp.where(winners, pot_total / num_winners, 0.0)
            payoffs = (chips_gained - new_chips_in.astype(jnp.float32)) / self.big_bet
            return payoffs

        rewards = lax.cond(
            game_done,
            compute_rewards,
            lambda: jnp.zeros(self.num_agents),
        )

        absorbing = jnp.broadcast_to(game_done, (self.num_agents,))
        done = absorbing.all()
        game_winner = (rewards > 0.0).astype(jnp.float32)
        info = {
            "returns": rewards.astype(jnp.float32),
            "timestep": next_state.timestep,
            "game_winner": game_winner,
        }
        return next_state, obs, rewards, absorbing, done, info

    def observation_space(self) -> Discrete:
        return Discrete(self.obs_dim)

    def action_space(self) -> Discrete:
        return Discrete(4)
