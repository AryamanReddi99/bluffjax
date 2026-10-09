"""
Bluff environment (AEC).

This code models the card game Bluff, also known as Cheat or I Doubt It:
https://en.wikipedia.org/wiki/Cheat_(game).

Rules
-----
num_decks decks of num_ranks x num_suits cards are dealt evenly; the
deck_size % num_agents cards left over start face down in the pile. A random
player leads. A claim is "size cards of rank r" (size 1 to 4, at most the
claimant's hand size): the claimant puts size cards from their hand face down
on the pile, of any ranks (they may lie). On a free lead (the first claim of
the game and the first claim after every challenge) the claimant names any
rank. Otherwise the rank is one above the previous claim's (cyclically).

After a claim the other players are asked in turn order, starting after the
claimant, whether to challenge it. The first to challenge turns the claimed
cards over: if the claim was a lie the claimant picks up the whole pile,
otherwise the challenger does, and the winner of the challenge makes the next
claim (a free lead). If nobody challenges, the next player in turn order makes
the next claim. A player wins when they have no cards left once their claim
has been resolved, i.e. it was not challenged or the challenge failed; a
player caught lying on their last cards picks up the pile and play goes on.
A game that reaches `horizon` steps is cut off without a winner (a draw).
This is a truncation, not an end of the game: `done` is set but `absorbing`
is not, so learners should bootstrap from the value of the state reached.

Steps
-----
Every step is one action of the player to act (`current_player_idx`). A claim
takes several steps of the same player, then one step per challenger:
    phase 3, free leads only: name the claimed rank (action = rank offset)
    phase 0: choose the claim size (action a: a + 1 cards)
    phase 1: pick the cards to play one at a time (action = rank offset of
             the card), claim size steps; the cards leave the hand together
             after the last pick
    phase 2: one step per other player in turn order (0 = challenge,
             1 = pass) until someone challenges or everyone has passed

Ranks are relative: rank-indexed actions and observation blocks are rotated so
that offset 0 is the reference rank `current_rank` (absolute rank =
(current_rank + offset) % num_ranks). Before the first claim of the game there
is no reference rank and offsets are absolute ranks. The reference rank is the
rank to claim in phase 0 (required, or the one just named), the claimed rank
in phases 1 and 2, and on a free lead after a challenge the rank that was
challenged. So in phase 1 offset 0 plays a card of the claimed rank.

Rewards (to the player concerned, on the step it happens):
    +1 per card to the claimant when nobody challenges the claim
    challenge: +1 to the winner, -1 per card in the pile to the player who
        picks it up
    +10 to the winner when the game ends

Observation (float32, for the player to act; seats are relative to that
player: seat k is the k-th player after them in turn order, seat 0 is
themselves; R = num_ranks, C = num_suits * num_decks cards per rank,
D = deck_size, n = num_agents):
    [0, CR)       own cards in hand, not counting the cards picked so far in
                  phase 1 (per rank thermometer, suit-major: index s * R + r
                  is 1 if holding more than s cards of rank offset r)
    +CR           cards picked so far in phase 1 (same encoding; 0 otherwise)
    +CR           claims on the pile since it was last picked up (same
                  encoding, capped at C per rank)
    +D            pile size (thermometer: index i is 1 if more than i cards)
    +nD           hand sizes of seats 0..n-1 (thermometer each). In phase 1
                  the acting player's size still includes the cards picked.
    +4            claim size in phases 1 and 2 (thermometer; 0 otherwise)
    +n            claimant's seat in phase 2 (one-hot; 0 otherwise)
    +4            phase one-hot (0 claim size, 1 play, 2 challenge, 3 name rank)
    +1            the claim being made is a free lead
    +CR           cards turned over at the most recent challenge (same
                  encoding as the hand; 0 before the first challenge)
    +n            seat of the player who picked up the pile at the most
                  recent challenge (one-hot; 0 before the first challenge)
    +1            the most recent challenge caught a lie
    total 4CR + (n + 1)D + 2n + 10 (378 for 2 players and 432 for 3 with one
    standard deck).
Everything except the own hand and the cards picked is public. Pile contents,
other hands and other players' picks are hidden.
"""

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from jax import lax

from bluffjax.utils.typing import BoolArray, FloatArray, IntArray, PRNGKeyArray
from bluffjax.environments.env import AECEnv
from bluffjax.environments.spaces import Discrete

# phases
CLAIM = 0  # choose the claim size
PLAY = 1  # pick the cards to play
CHALLENGE = 2  # other players challenge or pass
NAME_RANK = 3  # free lead: name the claimed rank
NUM_PHASES = 4

MAX_CLAIM = 4  # most cards in one claim


@struct.dataclass
class BluffState:
    pile_hand: FloatArray  # (R,) cards in the pile by rank
    pile_claims: FloatArray  # (R,) claimed cards in the pile by rank
    pile_size: IntArray
    agent_hands: FloatArray  # (n, R) cards by rank (incl. cards picked in phase 1)
    agent_hand_sizes: IntArray  # (n,)
    phase: IntArray  # CLAIM, PLAY, CHALLENGE or NAME_RANK
    current_player_idx: IntArray
    start_player_idx: IntArray  # first player of the game
    challenge_target_idx: IntArray  # claimant (in phases 0, 1, 3: the player to act)
    claim_size: IntArray  # 0 until chosen
    claim_rank: IntArray
    pending_play_hand: FloatArray  # (R,) cards picked (phase 1) / played (phase 2)
    pending_play_count: IntArray
    has_current_rank: BoolArray  # False until the first rank is named
    current_rank: IntArray  # reference rank of the relative rank offsets
    free_lead: BoolArray  # the claim being made is a free lead
    challenge_status: IntArray  # last resolution: 0 claim true, 1 lie caught, 2 not challenged, 3 claim in progress
    challenge_hand: FloatArray  # (R,) cards turned over at the most recent challenge
    challenge_loser_idx: IntArray  # picked up the pile at the most recent challenge (-1: none yet)
    challenge_lie: BoolArray  # the most recent challenge caught a lie
    absorbing: BoolArray  # (n,) the game ended (not set on truncation)
    done: bool  # the game ended or was truncated
    game_winner: BoolArray  # (n,) one-hot winner on the step the game ends, else 0
    timestep: int


class Bluff(AECEnv):
    def __init__(
        self,
        num_agents: int = 3,
        num_decks: int = 1,
        num_ranks: int = 13,
        num_suits: int = 4,
        horizon: int = 100,
    ) -> None:
        super().__init__(num_agents=num_agents, horizon=horizon)
        if num_agents < 2:
            raise ValueError(f"Bluff needs at least 2 players, got {num_agents}")
        self.num_ranks = num_ranks
        self.num_suits = num_suits
        self.cards_per_rank = num_suits * num_decks
        self.deck_size = num_suits * num_decks * num_ranks
        self.cards_per_deck = num_suits * num_ranks
        if self.deck_size < num_agents:
            raise ValueError("Bluff needs at least one card per player")
        # rank offsets (phases 1 and 3), claim sizes (phase 0), challenge/pass
        self.action_dim = max(num_ranks, MAX_CLAIM, 2)
        rank_block = self.cards_per_rank * num_ranks
        self.obs_dim = (
            4 * rank_block
            + (num_agents + 1) * self.deck_size
            + 2 * num_agents
            + MAX_CLAIM
            + NUM_PHASES
            + 2
        )

        self._reward_card = 1.0
        self._reward_challenge = 1.0
        self._reward_win = 10.0

    @partial(jax.jit, static_argnums=(0,))
    def _empty_hand(self) -> FloatArray:
        return jnp.zeros((self.num_ranks,), dtype=jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _encode_size_thermo(self, x: IntArray) -> FloatArray:
        return (jnp.arange(self.deck_size) < x).astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _card_ints_to_hand(self, cards: IntArray) -> FloatArray:
        ranks = cards % self.cards_per_deck // self.num_suits
        return jnp.bincount(ranks, length=self.num_ranks).astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _reference_rank(self, state: BluffState) -> IntArray:
        return jnp.where(state.has_current_rank, state.current_rank, 0)

    @partial(jax.jit, static_argnums=(0,))
    def _to_offsets(self, counts: FloatArray, state: BluffState) -> FloatArray:
        """Rank-indexed vector -> offset-indexed (offset 0 = reference rank)."""
        return jnp.roll(counts, -self._reference_rank(state), axis=0)

    @partial(jax.jit, static_argnums=(0,))
    def _offset_to_rank(self, state: BluffState, offset: IntArray) -> IntArray:
        return (self._reference_rank(state) + offset) % self.num_ranks

    @partial(jax.jit, static_argnums=(0,))
    def _encode_suit_major_thermo(self, counts: FloatArray) -> FloatArray:
        suit_ids = jnp.arange(self.cards_per_rank)[:, None]
        # suit-major layout: [suit0: R ranks, suit1: R ranks, ...]
        return (suit_ids < counts[None, :]).astype(jnp.float32).reshape(-1)

    @partial(jax.jit, static_argnums=(0,))
    def _encode_ranks(self, counts: FloatArray, state: BluffState) -> FloatArray:
        return self._encode_suit_major_thermo(self._to_offsets(counts, state))

    @partial(jax.jit, static_argnums=(0,))
    def _seat_one_hot(self, idx: IntArray, player_idx: IntArray) -> FloatArray:
        """One-hot of idx's seat relative to player_idx; zeros if idx < 0."""
        seat = (idx - player_idx) % self.num_agents
        return jax.nn.one_hot(seat, self.num_agents) * (idx >= 0)

    @partial(jax.jit, static_argnums=(0,))
    def _next_player(self, idx: IntArray) -> IntArray:
        return (idx + 1) % self.num_agents

    @partial(jax.jit, static_argnums=(0,))
    def obs_from_state(self, state: BluffState) -> FloatArray:
        player_idx = state.current_player_idx
        in_play = state.phase == PLAY
        in_challenge = state.phase == CHALLENGE
        # In phase 1 the cards picked so far belong to the player to act; in
        # phase 2 they are the claimant's hidden cards.
        picked = jnp.where(in_play, state.pending_play_hand, 0.0)
        own_hand = state.agent_hands[player_idx] - picked

        hand_sizes = jnp.roll(state.agent_hand_sizes, -player_idx)
        claim_shown = in_play | in_challenge
        claim_size_obs = (jnp.arange(MAX_CLAIM) < state.claim_size) & claim_shown
        claimant_obs = self._seat_one_hot(
            state.challenge_target_idx, player_idx
        ) * in_challenge
        return jnp.concatenate(
            [
                self._encode_ranks(own_hand, state),
                self._encode_ranks(picked, state),
                self._encode_ranks(state.pile_claims, state),
                self._encode_size_thermo(state.pile_size),
                jax.vmap(self._encode_size_thermo)(hand_sizes).reshape(-1),
                claim_size_obs.astype(jnp.float32),
                claimant_obs,
                jax.nn.one_hot(state.phase, NUM_PHASES, dtype=jnp.float32),
                state.free_lead.astype(jnp.float32)[None],
                self._encode_ranks(state.challenge_hand, state),
                self._seat_one_hot(state.challenge_loser_idx, player_idx),
                state.challenge_lie.astype(jnp.float32)[None],
            ],
            axis=0,
        ).astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: BluffState) -> BoolArray:
        action_ids = jnp.arange(self.action_dim)

        def claim_mask() -> BoolArray:
            hand_size = state.agent_hand_sizes[state.current_player_idx]
            return action_ids < jnp.minimum(MAX_CLAIM, hand_size)

        def play_mask() -> BoolArray:
            ranks = self._offset_to_rank(state, action_ids)
            available = (
                state.agent_hands[state.current_player_idx, ranks]
                - state.pending_play_hand[ranks]
            ) > 0
            return (
                available
                & (action_ids < self.num_ranks)
                & (state.pending_play_count < state.claim_size)
            )

        def challenge_mask() -> BoolArray:
            return action_ids < 2

        def name_rank_mask() -> BoolArray:
            return action_ids < self.num_ranks

        phase_masks = [claim_mask, play_mask, challenge_mask, name_rank_mask]
        return lax.cond(
            state.done,
            lambda: jnp.zeros((self.action_dim,), dtype=bool),
            lambda: lax.switch(state.phase, phase_masks),
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: PRNGKeyArray) -> tuple[BluffState, FloatArray]:
        rng_shuffle, rng_player = jax.random.split(rng)
        shuffled_deck = jax.random.permutation(rng_shuffle, self.deck_size)
        cards_per_player = self.deck_size // self.num_agents
        used_cards = self.num_agents * cards_per_player

        distributed = shuffled_deck[:used_cards].reshape(
            self.num_agents, cards_per_player
        )
        agent_hands = jax.vmap(self._card_ints_to_hand)(distributed)
        agent_hand_sizes = agent_hands.sum(axis=1).astype(jnp.int32)

        pile_cards_int = shuffled_deck[used_cards:]
        pile_hand = self._card_ints_to_hand(pile_cards_int)
        pile_size = pile_hand.sum().astype(jnp.int32)

        start_player = jax.random.randint(
            rng_player, shape=(), minval=0, maxval=self.num_agents
        ).astype(jnp.int32)

        state = BluffState(
            pile_hand=pile_hand,
            pile_claims=self._empty_hand(),
            pile_size=pile_size,
            agent_hands=agent_hands,
            agent_hand_sizes=agent_hand_sizes,
            phase=jnp.array(NAME_RANK, dtype=jnp.int32),
            current_player_idx=start_player,
            start_player_idx=start_player,
            challenge_target_idx=start_player,
            claim_size=jnp.array(0, dtype=jnp.int32),
            claim_rank=jnp.array(0, dtype=jnp.int32),
            pending_play_hand=self._empty_hand(),
            pending_play_count=jnp.array(0, dtype=jnp.int32),
            has_current_rank=jnp.array(False),
            current_rank=jnp.array(0, dtype=jnp.int32),
            free_lead=jnp.array(True),
            challenge_status=jnp.array(3, dtype=jnp.int32),
            challenge_hand=self._empty_hand(),
            challenge_loser_idx=jnp.array(-1, dtype=jnp.int32),
            challenge_lie=jnp.array(False),
            absorbing=jnp.zeros((self.num_agents,), dtype=bool),
            done=jnp.array(False),
            game_winner=jnp.zeros((self.num_agents,), dtype=bool),
            timestep=0,
        )
        obs = self.obs_from_state(state)
        return state, obs

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self, rng: PRNGKeyArray, state: BluffState, action: IntArray
    ) -> tuple[BluffState, FloatArray, FloatArray, BoolArray, bool, dict[str, Any]]:
        no_reward = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        action = jnp.asarray(action, dtype=jnp.int32)
        player = state.current_player_idx

        # Each phase returns (next state, reward, whether the claim was resolved).

        # phase 3: name the claimed rank (free lead)
        def do_name_rank() -> tuple[BluffState, FloatArray, BoolArray]:
            rank = self._offset_to_rank(state, action)
            next_state = state.replace(
                phase=jnp.array(CLAIM, dtype=jnp.int32),
                has_current_rank=jnp.array(True),
                current_rank=rank,
            )
            return next_state, no_reward, jnp.array(False)

        # phase 0: choose the claim size (1..MAX_CLAIM) of rank current_rank
        def do_claim() -> tuple[BluffState, FloatArray, BoolArray]:
            next_state = state.replace(
                phase=jnp.array(PLAY, dtype=jnp.int32),
                claim_size=jnp.minimum(action, MAX_CLAIM - 1) + 1,
                claim_rank=state.current_rank,
                pending_play_count=jnp.array(0, dtype=jnp.int32),
                pending_play_hand=self._empty_hand(),
                challenge_status=jnp.array(3, dtype=jnp.int32),
                challenge_target_idx=player,
            )
            return next_state, no_reward, jnp.array(False)

        # phase 1: pick the cards one at a time until claim_size are picked
        def do_play() -> tuple[BluffState, FloatArray, BoolArray]:
            chosen_rank = self._offset_to_rank(state, action)
            picked = state.pending_play_hand.at[chosen_rank].add(1.0)
            picked_count = state.pending_play_count + 1

            def finish_play() -> BluffState:
                next_hands = state.agent_hands.at[player].add(-picked)
                next_pile_hand = state.pile_hand + picked
                return state.replace(
                    agent_hands=next_hands,
                    agent_hand_sizes=next_hands.sum(axis=1).astype(jnp.int32),
                    pile_hand=next_pile_hand,
                    pile_claims=state.pile_claims.at[state.claim_rank].add(
                        state.claim_size.astype(jnp.float32)
                    ),
                    pile_size=next_pile_hand.sum().astype(jnp.int32),
                    phase=jnp.array(CHALLENGE, dtype=jnp.int32),
                    current_player_idx=self._next_player(player),
                    pending_play_hand=picked,
                    pending_play_count=picked_count,
                )

            def continue_play() -> BluffState:
                return state.replace(
                    pending_play_hand=picked, pending_play_count=picked_count
                )

            next_state = lax.cond(
                picked_count >= state.claim_size, finish_play, continue_play
            )
            return next_state, no_reward, jnp.array(False)

        # phase 2: the other players challenge or pass in turn order
        def do_challenge() -> tuple[BluffState, FloatArray, BoolArray]:
            target = state.challenge_target_idx
            challenger = player

            def resolve_challenge() -> tuple[BluffState, FloatArray, BoolArray]:
                claim_is_true = (
                    state.pending_play_hand[state.claim_rank] == state.claim_size
                )
                loser = jnp.where(claim_is_true, challenger, target)
                winner = jnp.where(claim_is_true, target, challenger)
                next_hands = state.agent_hands.at[loser].add(state.pile_hand)

                outcome_reward = no_reward.at[winner].add(self._reward_challenge)
                outcome_reward = outcome_reward.at[loser].add(
                    -state.pile_size.astype(jnp.float32) * self._reward_card
                )

                next_state = state.replace(
                    agent_hands=next_hands,
                    agent_hand_sizes=next_hands.sum(axis=1).astype(jnp.int32),
                    pile_hand=self._empty_hand(),
                    pile_claims=self._empty_hand(),
                    pile_size=jnp.array(0, dtype=jnp.int32),
                    phase=jnp.array(NAME_RANK, dtype=jnp.int32),
                    current_player_idx=winner,
                    challenge_target_idx=winner,
                    pending_play_hand=self._empty_hand(),
                    pending_play_count=jnp.array(0, dtype=jnp.int32),
                    claim_size=jnp.array(0, dtype=jnp.int32),
                    challenge_status=jnp.where(claim_is_true, 0, 1).astype(jnp.int32),
                    challenge_hand=state.pending_play_hand,
                    challenge_loser_idx=loser,
                    challenge_lie=~claim_is_true,
                    # offsets stay relative to the challenged rank until the
                    # winner names the next one
                    current_rank=state.claim_rank,
                    free_lead=jnp.array(True),
                )
                return next_state, outcome_reward, jnp.array(True)

            def pass_challenge() -> tuple[BluffState, FloatArray, BoolArray]:
                next_challenger = self._next_player(challenger)

                def finish_no_challenge() -> tuple[BluffState, FloatArray, BoolArray]:
                    next_player = self._next_player(target)
                    next_state = state.replace(
                        phase=jnp.array(CLAIM, dtype=jnp.int32),
                        current_player_idx=next_player,
                        challenge_target_idx=next_player,
                        pending_play_hand=self._empty_hand(),
                        pending_play_count=jnp.array(0, dtype=jnp.int32),
                        claim_size=jnp.array(0, dtype=jnp.int32),
                        challenge_status=jnp.array(2, dtype=jnp.int32),
                        current_rank=(state.claim_rank + 1) % self.num_ranks,
                        free_lead=jnp.array(False),
                    )
                    no_challenge_reward = no_reward.at[target].add(
                        state.claim_size.astype(jnp.float32) * self._reward_card
                    )
                    return next_state, no_challenge_reward, jnp.array(True)

                def continue_challenge() -> tuple[BluffState, FloatArray, BoolArray]:
                    next_state = state.replace(current_player_idx=next_challenger)
                    return next_state, no_reward, jnp.array(False)

                return lax.cond(
                    next_challenger == target, finish_no_challenge, continue_challenge
                )

            return lax.cond(action == 0, resolve_challenge, pass_challenge)

        next_state, reward, claim_resolved = lax.switch(
            state.phase, [do_claim, do_play, do_challenge, do_name_rank]
        )

        # The game ends when a claim is resolved and the claimant has no cards
        # left (a claimant caught lying has picked up the pile).
        empty_hand = next_state.agent_hand_sizes == 0
        game_over = claim_resolved & empty_hand.any()
        game_winner = empty_hand & game_over
        reward = reward + jnp.where(game_winner, self._reward_win, 0.0)

        next_timestep = state.timestep + 1
        truncated = (~game_over) & (next_timestep >= self.horizon)
        done = game_over | truncated
        absorbing = jnp.broadcast_to(game_over, (self.num_agents,))

        next_state = next_state.replace(
            timestep=next_timestep,
            game_winner=game_winner,
            done=done,
            absorbing=absorbing,
        )
        obs = self.obs_from_state(next_state)
        info = {
            "phase": next_state.phase,
            "claim_size": next_state.claim_size,
            "claim_rank": next_state.claim_rank,
            "challenge_status": next_state.challenge_status,
            "game_winner": game_winner,
            "truncated": truncated,
            "timestep": next_timestep,
        }
        return next_state, obs, reward, absorbing, done, info

    def observation_space(self) -> Discrete:
        return Discrete(self.obs_dim)

    def action_space(self) -> Discrete:
        return Discrete(self.action_dim)
