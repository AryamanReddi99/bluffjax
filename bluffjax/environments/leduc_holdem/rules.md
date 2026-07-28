# Leduc Hold'em — Rules

## Sources

- Southey, F., Bowling, M., Larson, B., Piccione, C., Burch, N., Billings, D., Rayner, C.
  "Bayes' Bluff: Opponent Modelling in Poker." *UAI 2005*.
  - https://poker.cs.ualberta.ca/publications/UAI05.pdf
  - https://arxiv.org/abs/1207.1411 
- RLCard — "Games in RLCard" documentation page (Leduc Hold'em section).
  https://rlcard.org/games.html
- OpenSpiel — `leduc_poker`
  https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/games/leduc_poker/leduc_poker.h


## Objective

Leduc Hold'em is a small zero-sum imperfect-information poker
game used as a tractable testbed for game-theoretic and RL algorithms (best-response/CFR
solvers, exploitability computation, etc.) — small enough to solve exactly,
but retaining hold'em-style structure (private cards, a public card, multiple betting rounds,
raise caps). Each player tries to maximize their expected net chip winnings over a hand.

## Setup

- **Deck**: 6 cards total — 2 suits x 3 ranks (canonically Jack, Queen, King). Suits are
  irrelevant to hand strength; only rank matters.
- **Players**: exactly 2 (heads-up).
- **Ante**: both players post 1 chip before round 1 (no separate blinds).
- **Dealing**: each player is dealt one private ("hole") card face-down. A third card is
  held back to be revealed as the single public/community card at the start of round 2.

## Gameplay

- **Two betting rounds**:
  1. **Round 1 (pre-flop)**: after antes and hole cards are dealt, betting begins.
  2. **Round 2 (post-flop)**: one public card is revealed face-up (shared by both players),
     then a second betting round occurs.
- **Actions** (3 total): **Fold**, **Call** (includes "check" when there is nothing extra to
  match), **Raise**.
- **Raise cap**: a fixed maximum of raises per round (a "two-bet maximum" — at most 2 raises
  per round in the canonical game).
- **Raise sizes**: fixed per round — a smaller fixed increment in round 1, a larger one in
  round 2 (canonically 2 and 4 chips respectively).
- **Betting closes** a round once all active players have matched the current bet level
  (either by all checking around, or by everyone calling the last raise).
- **Showdown / hand ranking** (if neither player folds): each player's hand is their private
  card plus the public card.
  - **Pair** (private card's rank equals the public card's rank) beats **any** high-card hand.
  - Otherwise, hands are compared by the higher of the two cards' rank (King > Queen > Jack);
    ties split the pot.
  - If a player folds, the remaining player wins the entire pot uncontested.

## Environment Rules

- Player 0 always acts first in **both** betting rounds.
- **Fold is only a legal action when facing a bet**.
  **Call is always legal.** **Raise is legal iff `num_raises < MAX_RAISES` (i.e. 0 or 1 raises
  already made this round)**.
- The hand (episode) ends when either a player has folded (`remaining_after <= 1`) or round 2's
  betting closes (`stage == 2 and round_over`).
- **Payoffs**: at hand end, the pot (`sum` of both players' cumulative `ante` contributions) is
  split among winners and reward for each player is
  `final_money - STARTING_MONEY`, i.e. net chip change for the hand - zero-sum by construction.
- **Observation** (`obs_from_state`, 36-dim one-hot vector, built from the *current* player's
  perspective): indices `0-2` = current player's own hole-card rank one-hot (J/Q/K); indices
  `3-5` = public-card rank one-hot (all zero if not yet revealed); indices `6-20` = current
  player's own cumulative ante, one-hot, clipped to a max index of 14 (i.e. ante amounts above
  14 chips are all encoded at the same slot); indices `21-35` = opponent's cumulative ante,
  same one-hot/clip scheme.

## Simplifications

Similar to Kuhn Poker, there is little room for simplification in Leduc Poker.
