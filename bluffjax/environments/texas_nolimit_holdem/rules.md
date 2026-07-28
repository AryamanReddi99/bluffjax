# Texas No-Limit Hold'em - Rules


## Sources

- Gilpin, A. & Sandholm, T., ["A heads-up no-limit Texas Hold'em poker player: discretized
  betting models and automatically generated equilibrium-finding programs"](https://www.cs.cmu.edu/~sandholm/tartanian.AAMAS08.pdf),
  AAMAS 2008.
- Brown, N. & Sandholm, T., ["Superhuman AI for multiplayer poker"](https://www.science.org/doi/10.1126/science.aay2400),
  Science, 2019.

## Objective

Win chips from your opponent(s), either by having the best five-card hand at showdown or
by getting all other players to fold before showdown.

## Setup

- **Deck**: standard 52-card deck, no jokers, shuffled fresh every hand.
- **Hole cards**: each player is dealt 2 private cards.
- **Community cards**: 5 shared cards, revealed in three stages (flop 3, turn 1, river 1).
- **Blinds**: two forced bets before cards are even looked at - a small blind and a big
  blind (typically double the small blind), posted by the two players seated after the
  dealer button. They rotate every hand.
- **Starting stack**: every player buys in for the same number of chips at the start of
  the hand ("table stakes" - you can't bet more than what's in front of you).

## Gameplay

A hand proceeds through up to four betting rounds ("streets"):

1. **Pre-flop**: hole cards dealt, blinds posted, betting starts with the player after
   the big blind (heads-up: the small blind/button acts first).
2. **Flop**: 3 community cards revealed, betting starts with the first active player
   after the button.
3. **Turn**: 1 more community card revealed, another betting round.
4. **River**: 1 final community card revealed, final betting round.

On each betting round a player facing a bet may **fold**, **call** (match the bet), or
**raise** (increase the bet); a player facing no bet may **check** (pass) or **bet**. In
**no-limit** betting specifically, a player may bet or raise **any amount from the table
minimum raise up to the entirety of their remaining stack** - there is no upper limit
imposed by the pot size (that would be pot-limit) or by fixed increments (that would be
limit hold'em). Going all-in for less than a full call/raise is handled via the
table-stakes rule.

If more than one player remains after the river betting round, there is a **showdown**:
each player makes the best five-card hand from their 2 hole cards + 5 community cards
(standard poker hand rankings, high card through royal flush), and the best hand(s) win
the pot, split evenly among ties.

## Environment Rules

- The full deck is shuffled once per hand in `reset()`; both players' hole cards and
  **all five** community cards (flop, turn, river) are dealt up front. The `stage` field
  only controls how many community cards are *revealed in the observation*
  (dims 0-51 are a one-hot over the 52 cards for own hole cards + the
  community cards visible at the current stage); the actual cards used for showdown are
  fixed from the start of the hand. This means an early all-in is automatically "run out"
  to a full 5-card board with no special-case logic needed.
- Which player posts the small blind is drawn uniformly at random every `reset()`; there is no persistent button rotation
  across hands, because hands do not share state.
- Pre-flop, the small blind acts first.
- A betting round ends once (consecutive checks/calls) reaches the number of players who are still active and not already all-in.
- `obs_dim = 54`: dims 0-51 one-hot own hole cards + currently-visible community cards,
  dim 52 = current player's own total chips committed this hand, dim 53 = the maximum
  chips committed by any player this hand.
- **Action space, `num_actions = 5`**. Let
  `pot = sum(chips_in)` be the **total chips already committed by all players this
  hand**, `half_pot = floor(pot / 2)`, `max_round` the
  largest per-street contribution so far, `diff = max_round - player_round` the amount
  needed to call, and `player_remain` the acting player's remaining stack:
  - **0 - check/call**: adds exactly `diff` chips to the pot (0 if nothing to call).
    Always legal for an active player (not folded, not all-in, has chips left).
  - **1 - raise half pot**: adds exactly `half_pot = floor(pot/2)` chips to the pot -
    this flat amount *is* the player's whole new contribution for the action (it is not
    "call `diff` then raise by half a pot on top"). Legal only if the player has enough
    chips (`half_pot <= player_remain`) **and** `half_pot` actually exceeds what's needed
    to call (`half_pot + player_round > max_round`), i.e. it must be a genuine raise, not
    a tie/under-call.
  - **2 - raise pot**: same mechanism, but adds `pot` chips (the entire current pot size)
    as the new contribution. Legal if `pot <= player_remain` (and implicitly always
    exceeds the call amount once any chips are in the pot, so no separate "is this a
    raise" check is coded for this action).
  - **3 - all-in**: adds the player's entire `player_remain` to the pot, zeroing their
    stack and flagging them all-in. Per the legality mask, this action is only offered
    when the player has *more* than enough to call (`diff < player_remain`) - i.e. it
    represents "shove for a raise," not "call for less."
  - **4 - fold**: forfeits the hand; no chips move.
- **Payoffs**: computed once `game_done` - winners are determined by comparing best
  5-card-from-7 hand scores (`_compare_hands` in `poker_utils.py`, evaluated over all 21
  five-card combinations of the 7 cards) among non-folded players, or trivially the lone
  remaining player if everyone else folded. The total pot is split evenly among winners;
  each player's reward is `chips_won - chips_they_put_in`.

## Simplifications

- **Real no-limit hold'em allows a continuous range of bet sizes** (any amount from the minimum raise up to your full stack), whereas this environment **discretizes betting into exactly 5 fixed actions**:
check/call, a half-pot-sized bet, a pot-sized bet, all-in, or fold. This is a classic
"bet abstraction" used throughout poker AI research to shrink an effectively
infinite action space down to a small, tractable menu - but it means an agent in this
environment can never, e.g., raise to exactly 3x the big blind, min-raise, or bet 75% pot;
its only sizing choices at any decision point are "half the current pot," "the whole
current pot," or "everything."
- **No true side-pot accounting.**
- **No persistent bankroll / tournament structure.** `reset()` always reinitializes both
  players to exactly `init_chips = 100` minus blinds; nothing in the state carries over
  between hands. Each hand is an independent, equal-stack episode - there is no chip accumulation, elimination, or re-buy logic across hands.
- **Random, non-alternating blind assignment.** Which player posts the small blind is redrawn uniformly at random every hand rather than rotating
  deterministically to the next player as in a real cardroom or tournament.
