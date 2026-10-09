"""
5-Card Draw poker environment.

One hand of no-limit 5-card draw for 2-10 players with equal stacks
(init_chips) and blinds of 1 and 2. The small blind is drawn at random every
hand, so player indices are only an internal seat order.

- Pre-draw betting starts with the player after the big blind (heads-up: the
  small blind).
- Draw: every player still in the hand, all-in players included, keeps any
  subset of their 5 cards and draws replacements from the stock, one player
  at a time starting with the first player after the dealer. Heads-up the
  small blind is the dealer; with 3 or more players the dealer sits just
  before the small blind. When the stock runs out, the discards of the players
  who drew earlier are shuffled into a new stock. The drawing player's own
  discards are shuffled in only if that is still too few cards, which can
  happen only with 10 players (2 cards left after the deal).
- Post-draw betting starts with the first player after the dealer who can
  still bet. If fewer than two players can bet (the others are all-in), the
  hand goes to showdown right after the draw.
- Betting actions: check/call, raise half the pot, raise the pot, all-in,
  fold. A raise puts in floor(pot / 2), pot or all remaining chips; the legal
  ones are exactly those that raise the highest bet. A betting round ends
  once every player who can still bet has acted since the last raise.
- Rewards are paid when the hand ends: chips won minus chips put in. Ties
  split the pot.
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
from bluffjax.utils.game_utils.poker_utils import _compare_five_card_hands

DECK_SIZE = 52
HAND_SIZE = 5
MAX_PLAYERS = DECK_SIZE // HAND_SIZE  # 10
FOLD = 4
NUM_BETTING_ACTIONS = 5
NUM_DRAW_ACTIONS = 2**HAND_SIZE


@struct.dataclass
class FiveCardDrawState:
    agent_cards: IntArray  # (num_agents, 5) each player's hand, sorted
    chips_in: FloatArray  # (num_agents,) chips put in the pot this hand
    round_raised: FloatArray  # (num_agents,) chips put in this betting round
    remaining_chips: FloatArray  # (num_agents,) chips behind
    small_blind_idx: IntArray
    current_player_idx: IntArray
    stage: IntArray  # 0=pre-draw betting, 1=draw, 2=post-draw betting
    folded: BoolArray
    all_in: BoolArray
    not_raise_num: IntArray  # players able to bet who acted since the last raise
    shuffled_deck: IntArray  # (52,) the stock is shuffled_deck[deck_idx:]
    deck_idx: IntArray
    discards: BoolArray  # (52,) card is in the discard pile
    draw_start_idx: IntArray  # first player to draw (detects the end of the draw)
    absorbing: BoolArray
    done: bool
    timestep: int


class FiveCardDraw(AECEnv):
    def __init__(
        self, num_agents: int = 2, horizon: int = 100_000, init_chips: int = 100
    ) -> None:
        if not 2 <= num_agents <= MAX_PLAYERS:
            raise ValueError(
                f"5-Card Draw needs 2 to {MAX_PLAYERS} players, got {num_agents}"
            )
        super().__init__(num_agents=num_agents, horizon=horizon)
        self.deck_size = DECK_SIZE
        self.small_blind = 1
        self.big_blind = 2 * self.small_blind
        if init_chips <= self.big_blind:
            raise ValueError(
                f"init_chips must exceed the big blind ({self.big_blind}), "
                f"got {init_chips}"
            )
        self.init_chips = init_chips
        self.obs_dim = self.deck_size + 2
        # 0-4 betting, 5-36 draw (2^5 keep/discard patterns)
        self.num_actions = NUM_BETTING_ACTIONS + NUM_DRAW_ACTIONS

    def observation_space(self) -> Discrete:
        return Discrete(self.obs_dim)

    def action_space(self) -> Discrete:
        return Discrete(self.num_actions)

    @partial(jax.jit, static_argnums=(0,))
    def obs_from_state(self, state: FiveCardDrawState) -> FloatArray:
        """
        Observation format (54 dimensions) - same as no-limit:
        - 0-51: One-hot encoding of current player's 5 cards
        - 52: Current player's chips in pot
        - 53: Max chips in pot
        """
        hand_cards = state.agent_cards[state.current_player_idx]
        obs = jnp.zeros(self.obs_dim, dtype=jnp.float32)
        obs = obs.at[hand_cards].set(1.0)
        obs = obs.at[52].set(
            state.chips_in[state.current_player_idx].astype(jnp.float32)
        )
        obs = obs.at[53].set(jnp.max(state.chips_in).astype(jnp.float32))
        return obs

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: FiveCardDrawState) -> BoolArray:
        """
        Actions: 0-4 betting (check/call, raise half pot, raise pot, all in, fold),
                5-36 draw: action 5+k = binary pattern k for keep/discard.
                Bit j=0 (LSB) = card at position 0 (smallest), bit 4 = position 4 (largest).
                0=discard, 1=keep. So action 5 = [00000] = discard all, action 36 = [11111] = keep all.
        Every raise that is legal raises the highest bet. Nothing is legal once
        the hand is over.
        """
        avail_actions = jnp.zeros(self.num_actions, dtype=bool)
        in_play = ~state.done

        # Betting stages (0 and 2)
        is_betting = in_play & ((state.stage == 0) | (state.stage == 2))
        current_player = state.current_player_idx
        player_round = state.round_raised[current_player]
        max_round = jnp.max(state.round_raised)
        player_remain = state.remaining_chips[current_player]
        pot = jnp.sum(state.chips_in)
        half_pot = jnp.floor(pot / 2.0)
        diff = max_round - player_round

        can_raise = player_remain > diff
        can_raise_pot = can_raise & (pot <= player_remain)
        can_raise_half = (
            can_raise
            & (half_pot <= player_remain)
            & ((half_pot + player_round) > max_round)
        )

        avail_actions = avail_actions.at[0].set(is_betting)
        avail_actions = avail_actions.at[1].set(is_betting & can_raise_half)
        avail_actions = avail_actions.at[2].set(is_betting & can_raise_pot)
        avail_actions = avail_actions.at[3].set(is_betting & can_raise)
        avail_actions = avail_actions.at[FOLD].set(is_betting)

        # Draw stage (stage 1) - all 32 binary patterns valid for current player
        in_draw = in_play & (state.stage == 1) & ~state.folded[current_player]
        avail_actions = avail_actions.at[NUM_BETTING_ACTIONS:].set(in_draw)

        return avail_actions

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: PRNGKeyArray) -> tuple[FiveCardDrawState, FloatArray]:
        """Return the first state of the game"""
        rng_shuffle, rng_player = jax.random.split(rng)
        shuffled_deck = jax.random.permutation(rng_shuffle, self.deck_size)
        agent_cards = jnp.sort(
            shuffled_deck[: HAND_SIZE * self.num_agents].reshape(-1, HAND_SIZE), axis=1
        )
        deck_idx = jnp.int32(HAND_SIZE * self.num_agents)

        small_blind_agent = jax.random.randint(
            rng_player, shape=(), minval=0, maxval=self.num_agents
        )
        big_blind_agent = (small_blind_agent + 1) % self.num_agents
        chips_in = jnp.zeros(self.num_agents)
        chips_in = chips_in.at[small_blind_agent].set(self.small_blind)
        chips_in = chips_in.at[big_blind_agent].set(self.big_blind)
        round_raised = chips_in
        remaining_chips = jnp.full(self.num_agents, self.init_chips) - chips_in
        current_player_idx = (small_blind_agent + 2) % self.num_agents

        state = FiveCardDrawState(
            agent_cards=agent_cards,
            chips_in=chips_in,
            round_raised=round_raised,
            remaining_chips=remaining_chips,
            small_blind_idx=small_blind_agent,
            current_player_idx=current_player_idx,
            stage=jnp.int32(0),
            folded=jnp.zeros(self.num_agents, dtype=bool),
            all_in=jnp.zeros(self.num_agents, dtype=bool),
            not_raise_num=jnp.int32(0),
            shuffled_deck=shuffled_deck,
            deck_idx=deck_idx,
            discards=jnp.zeros(self.deck_size, dtype=bool),
            draw_start_idx=jnp.int32(0),
            absorbing=jnp.zeros(self.num_agents, dtype=bool),
            done=jnp.array(False),
            timestep=0,
        )

        obs = self.obs_from_state(state)
        return state, obs

    def _next_eligible(self, start_idx: IntArray, eligible: BoolArray) -> IntArray:
        """First eligible player after start_idx in seat order (start_idx if none)."""
        idxs = (start_idx + jnp.arange(1, self.num_agents + 1)) % self.num_agents
        ok = eligible[idxs]
        return jnp.where(ok.any(), idxs[jnp.argmax(ok)], start_idx)

    def _dealer_idx(self, small_blind_idx: IntArray) -> IntArray:
        """Heads-up the small blind is the dealer; otherwise the player before it."""
        if self.num_agents == 2:
            return small_blind_idx
        return (small_blind_idx - 1) % self.num_agents

    def _draw(
        self,
        state: FiveCardDrawState,
        player: IntArray,
        keep: BoolArray,
        rng: PRNGKeyArray,
    ) -> tuple[IntArray, IntArray, IntArray, BoolArray]:
        """player discards the cards of its sorted hand where keep is False and
        draws as many from the stock; if the stock runs out, the discard pile is
        shuffled into a new stock (with player's own discards only if the pile
        is too small). Returns agent_cards, shuffled_deck, deck_idx, discards."""
        hand = state.agent_cards[player]
        num_discard = HAND_SIZE - jnp.sum(keep).astype(jnp.int32)
        own = jnp.zeros(self.deck_size, dtype=bool).at[hand].set(~keep)
        from_stock = jnp.minimum(num_discard, self.deck_size - state.deck_idx)
        need = num_discard - from_stock
        use_own = need > jnp.sum(state.discards)
        pool = state.discards | (own & use_own)
        pool_size = jnp.sum(pool).astype(jnp.int32)
        # New stock: the pool in random order, at the end of the deck array.
        priority = jax.random.permutation(rng, self.deck_size)
        order = jnp.argsort(jnp.where(pool, self.deck_size + priority, priority))
        pos = jnp.arange(HAND_SIZE)
        last = self.deck_size - 1
        stock_cards = state.shuffled_deck[jnp.minimum(state.deck_idx + pos, last)]
        pool_cards = order[
            jnp.clip(self.deck_size - pool_size + pos - from_stock, 0, last)
        ]
        drawn = jnp.where(pos < from_stock, stock_cards, pool_cards)
        kept = jnp.sort(jnp.where(keep, hand, self.deck_size))  # kept cards first
        num_kept = HAND_SIZE - num_discard
        new_hand = jnp.where(
            pos < num_kept, kept, drawn[jnp.clip(pos - num_kept, 0, HAND_SIZE - 1)]
        )
        reshuffled = need > 0
        shuffled_deck = jnp.where(reshuffled, order, state.shuffled_deck)
        deck_idx = jnp.where(
            reshuffled,
            self.deck_size - pool_size + need,
            state.deck_idx + num_discard,
        ).astype(jnp.int32)
        discards = jnp.where(reshuffled, own & ~use_own, state.discards | own)
        agent_cards = state.agent_cards.at[player].set(jnp.sort(new_hand))
        return agent_cards, shuffled_deck, deck_idx, discards

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self, rng: PRNGKeyArray, state: FiveCardDrawState, action: IntArray
    ) -> tuple[
        FiveCardDrawState, FloatArray, FloatArray, BoolArray, bool, dict[str, Any]
    ]:
        """
        Execute one step. Actions: 0-4 betting, 5-36 draw (2^5 binary keep/discard).
        The action must be legal (get_avail_actions).
        """
        current_player = state.current_player_idx
        stage = state.stage
        is_betting = stage != 1
        is_draw = stage == 1

        # --- Betting (stages 0 and 2) ---
        bet_action = jnp.minimum(action, FOLD)
        player_round = state.round_raised[current_player]
        max_round = jnp.max(state.round_raised)
        player_remain = state.remaining_chips[current_player]
        pot = jnp.sum(state.chips_in)
        half_pot = jnp.floor(pot / 2.0)
        diff = max_round - player_round
        bet_amounts = jnp.stack(
            [diff, half_pot, pot, player_remain, jnp.zeros_like(pot)]
        )
        amount = jnp.where(is_betting, bet_amounts[bet_action], 0.0)
        is_fold = is_betting & (bet_action == FOLD)
        new_chips_in = state.chips_in.at[current_player].add(amount)
        new_round_raised = state.round_raised.at[current_player].add(amount)
        new_remaining_chips = state.remaining_chips.at[current_player].add(-amount)
        went_all_in = (
            is_betting & ~is_fold & (new_remaining_chips[current_player] <= 0)
        )
        new_all_in = state.all_in.at[current_player].set(
            state.all_in[current_player] | went_all_in
        )
        new_folded = state.folded.at[current_player].set(
            state.folded[current_player] | is_fold
        )
        # Every legal raise raises the highest bet, so everyone else who can
        # still bet must respond. The raiser counts as having acted unless it
        # is all-in (then it can't act again). A call counts unless it puts
        # the caller all-in; a fold leaves the count unchanged.
        is_raise = is_betting & (bet_action >= 1) & (bet_action <= 3)
        is_call = is_betting & (bet_action == 0)
        new_not_raise_num = jnp.where(
            is_raise,
            jnp.where(went_all_in, 0, 1),
            jnp.where(
                is_call & ~went_all_in,
                state.not_raise_num + 1,
                state.not_raise_num,
            ),
        ).astype(jnp.int32)

        num_active = jnp.sum(~new_folded)
        can_bet = ~new_folded & ~new_all_in
        num_playable = jnp.sum(can_bet)
        round_over = is_betting & (new_not_raise_num >= num_playable)
        fold_win = is_betting & (num_active <= 1)

        # --- Draw (stage 1) ---
        draw_pattern = jnp.clip(action - NUM_BETTING_ACTIONS, 0, NUM_DRAW_ACTIONS - 1)
        keep = ((draw_pattern >> jnp.arange(HAND_SIZE)) & 1).astype(bool)
        did_draw = is_draw & (action >= NUM_BETTING_ACTIONS) & ~state.folded[
            current_player
        ]
        drawn_cards, drawn_deck, drawn_deck_idx, drawn_discards = self._draw(
            state, current_player, keep, rng
        )
        new_agent_cards = jnp.where(did_draw, drawn_cards, state.agent_cards)
        new_shuffled_deck = jnp.where(did_draw, drawn_deck, state.shuffled_deck)
        new_deck_idx = jnp.where(did_draw, drawn_deck_idx, state.deck_idx)
        new_discards = jnp.where(did_draw, drawn_discards, state.discards)
        next_drawer = self._next_eligible(current_player, ~state.folded)
        draw_over = did_draw & (next_drawer == state.draw_start_idx)

        # --- Transitions ---
        dealer = self._dealer_idx(state.small_blind_idx)
        first_drawer = self._next_eligible(dealer, ~new_folded)
        first_bettor = self._next_eligible(dealer, can_bet)
        next_bettor = self._next_eligible(current_player, can_bet)

        to_draw = (stage == 0) & round_over & ~fold_win
        to_post_draw_betting = draw_over & (num_playable >= 2)
        showdown_after_draw = draw_over & (num_playable < 2)
        game_done = fold_win | ((stage == 2) & round_over) | showdown_after_draw

        new_stage = jnp.where(
            to_draw,
            jnp.int32(1),
            jnp.where(to_post_draw_betting, jnp.int32(2), stage),
        )
        new_current_player_idx = current_player
        new_current_player_idx = jnp.where(
            is_betting & ~round_over, next_bettor, new_current_player_idx
        )
        new_current_player_idx = jnp.where(
            to_draw, first_drawer, new_current_player_idx
        )
        new_current_player_idx = jnp.where(
            did_draw & ~draw_over, next_drawer, new_current_player_idx
        )
        new_current_player_idx = jnp.where(
            to_post_draw_betting, first_bettor, new_current_player_idx
        )
        new_draw_start_idx = jnp.where(to_draw, first_drawer, state.draw_start_idx)

        # A new betting round starts from zero.
        new_round_raised = jnp.where(
            round_over, jnp.zeros_like(new_round_raised), new_round_raised
        )
        new_not_raise_num = jnp.where(round_over, jnp.int32(0), new_not_raise_num)

        def compute_rewards():
            winners = _compare_five_card_hands(new_agent_cards, new_folded)
            winners = jnp.where(num_active <= 1, ~new_folded, winners)
            pot_total = jnp.sum(new_chips_in).astype(jnp.float32)
            num_winners = jnp.sum(winners).astype(jnp.float32)
            chips_gained = jnp.where(winners, pot_total / num_winners, 0.0)
            return chips_gained - new_chips_in.astype(jnp.float32)

        rewards = lax.cond(
            game_done,
            compute_rewards,
            lambda: jnp.zeros(self.num_agents),
        )

        absorbing = jnp.broadcast_to(game_done, (self.num_agents,))
        next_state = FiveCardDrawState(
            agent_cards=new_agent_cards,
            chips_in=new_chips_in,
            round_raised=new_round_raised,
            remaining_chips=new_remaining_chips,
            small_blind_idx=state.small_blind_idx,
            current_player_idx=new_current_player_idx,
            stage=new_stage,
            folded=new_folded,
            all_in=new_all_in,
            not_raise_num=new_not_raise_num,
            shuffled_deck=new_shuffled_deck,
            deck_idx=new_deck_idx,
            discards=new_discards,
            draw_start_idx=new_draw_start_idx,
            absorbing=absorbing,
            done=game_done,
            timestep=state.timestep + 1,
        )

        obs = self.obs_from_state(next_state)
        done = absorbing.all()
        game_winner = (rewards > 0.0).astype(jnp.float32)
        info = {
            "returns": rewards.astype(jnp.float32),
            "timestep": next_state.timestep,
            "game_winner": game_winner,
        }
        return next_state, obs, rewards, absorbing, done, info
