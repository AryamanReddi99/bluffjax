# Goofspiel (Game of Pure Strategy / GOPS)

## Sources

- Ross, S. M. (1971).
  ["Goofspiel — the game of pure strategy."](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/CB85A022644C516D0D59BFAEF91B69AE/S0021900200035725a.pdf/goofspiel_the_game_of_pure_strategy.pdf)
  *Journal of Applied Probability*, 8(3), 621–625.
- OpenSpiel source,
  [`open_spiel/games/goofspiel/goofspiel.h`](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/games/goofspiel/goofspiel.h)


## Objective

Each player tries to accumulate the highest total value of "prize" cards by
winning them through simultaneous bids made with cards from their
own hand. The player with the most accumulated points at the end of the
game wins.

## Setup

A standard 52-card deck is split into three 13-card suits:

- one suit is the shuffled, face-down prize pile, revealed one card per
  round;
- the other two suits are dealt one each to the two bidding players.


## Gameplay

1. **Prize reveal**: at the start of each round, the next card of the
   (shuffled) prize deck is turned face up. Its rank (1–13, Ace low, King
   high) is its point value.
2. **Simultaneous bidding**: every player privately selects one card from
   their own hand (also ranked 1–13); both players reveal their chosen bid
   card at the same time.
3. **Resolving the round**: the player with the strictly highest bid card
   wins the prize card and scores points equal to its rank. If two or
   more players tie for the highest bid, the prize card is **discarded** (no one gets it).
4. **Card consumption**: each bid card can only be used once — once
   played, it is removed from that player's hand for the rest of the
   game.
5. **Game length**: the game lasts 13 rounds (one round per prize card /
   one per card in each player's starting hand).
6. **Scoring/winner**: at the end of 13 rounds, whoever has accumulated
   the most total prize-card value wins (ties for the overall game are
   possible if final point totals are equal).

## Environment Rules

### Default configuration: `num_ranks=13`

- **Actions**: `num_actions = deck_size = 13`; an action is the index
  (0–12) of the card a player bids from their own hand.
  `get_avail_actions` masks out cards already played, and masks
  everything once `state.done` is `True`.
- **Prize order — shuffled/random each episode** (matches Openspiel's implementation).
- **Prize value**:
  `prize_value = (state.deck[state.current_round] + 1)`,
  i.e. the revealed prize card's rank is `1..13` (converting the
  0-indexed permutation entry to a 1–13 point value).
- **Observation** dim `deck_size * 3 = 39` per agent
  is the concatenation of:
  - a one-hot of the current round's prize card rank (13 dims);
  - the observing agent's own played/used cards, as a binary vector (13
    dims);
  - one other agent's played/used cards, as a binary vector (13 dims)


## Simplifications vs. Standard/Other-Cited Goofspiel Variants


- Hardcoded to 2 players. This deviates from the ossible three-player variant
  described by Ross.
- **Prize order is randomized every episode - this matches OpenSpiel's default `points_order="random"` but
  is a deliberate simplification relative to the "ascending"/
  "descending" fixed-order variants also supported by OpenSpiel and
  discussed in the literature.
- **Ties discard the prize rather than splitting its value.**
- **Not strictly zero-sum / constant-sum.** 

