# Five Card Draw - Rules

## Sources

- [Pagat.com: Five Card Draw Poker](https://www.pagat.com/poker/variants/5draw.html) - fetched 2026-07-27
- [PokerStars: Five Card Draw](https://www.pokerstars.com/poker/games/draw/) - fetched 2026-07-27


## Objective

Win chips by holding the best five-card poker hand at showdown, or by being
the last player left after all opponents fold.

## Setup

- Standard 52-card deck, no jokers/wild cards.
- Traditionally any number of players; modern casino/online versions (e.g.
  PokerStars) are typically short-handed with a dealer button and blinds.
- Forced bets: home games use an ante from every player; casino/online play
  uses a small blind + big blind posted by the two seats left of the button.
- Each player is dealt five private cards, face down, one at a time.

## Gameplay

1. **Pre-draw betting round.** Action starts left of the big blind (or the
   dealer, in ante games). Players fold, check/call, or bet/raise in turn.
   The round ends once all live players have matched the current bet.
2. **The draw.** Starting left of the button, each player discards some
   number of cards and receives that many replacements from the deck. A
   player may discard zero ("standing pat"). House rules vary: many home
   games cap discards at three (four if showing an ace); other rule sets
   (e.g. PokerStars) allow discarding and replacing all five. If the deck
   runs short, discards are reshuffled to continue dealing.
3. **Post-draw betting round.** A second round follows the draw, again
   starting from the first live player after the button; betting limits
   often double vs. round one in fixed-limit games.
4. **Showdown.** Remaining players reveal hands (usually starting with the
   last aggressor); best hand wins, ties split the pot equally. If only one
   player remains at any point, they win the pot uncontested, no showdown.
5. **Standard hand ranking** (best to worst): straight flush (royal flush =
   ace-high case), four of a kind, full house, flush, straight, three of a
   kind, two pair, one pair, high card. Can be played limit, pot-limit, or
   no-limit depending on house rules.

## Environment Rules


**Parameters:** `init_chips=100` starting stack per player, `horizon=100_000` (a hand
ends well before this via fold/showdown). One hand per episode; there is no multi-hand/bankroll loop in the env.

The
small-blind seat is chosen uniformly at random each episode; the other seat posts big blind, both amounts credited automatically to `chips_in` /
`round_raised`. With `num_agents=2`,
`current_player_idx = (small_blind_idx + 2) % num_agents` reduces to the
small-blind seat, so the small blind acts first pre-draw and the big blind
acts first post-draw - standard heads-up structure.

**Stages** (`state.stage`): `0` = pre-draw betting, `1` = draw, `2` =
post-draw betting. A betting round ends once a running counter of
consecutive non-raising actions (`not_raise_num`) reaches the number of
players still able to act (not folded, not all-in).

**Action space is `Discrete(37)`:**

- **Actions 0-4 (betting; legal only in stage 0 or 2):**
  - `0` - **Check / Call.** Adds `diff = max_round − player_round`. If
    `diff == 0` this is a check; otherwise it calls exactly. Always legal.
  - `1` - **Raise half-pot.** Adds `floor(pot / 2)` chips, where
    `pot = sum(chips_in)` is the total put in by all players this whole
    hand (not just the round). Available only if it fits the player's
    stack and actually exceeds a call (`half_pot + player_round >
    max_round`) - it must be a genuine raise, not just a call.
  - `2` - **Raise pot.** Adds `pot` chips. Available if `pot <= remaining
    chips`.
  - `3` - **All-in.** Adds the player's entire remaining stack. Available
    whenever remaining chips exceed `diff`.
  - `4` - **Fold.** Always legal; marks the player folded.

  Actions 1-3 add a fixed pot- or stack-relative amount *on top of* the
  player's current round contribution - "add X more chips", not "raise to
  X" - and the raise menu is fixed to exactly these three sizes (§6).

- **Actions 5-36 (draw; legal only in stage 1, for a non-folded player):**
  Each encodes one of 2⁵ = 32 keep/discard patterns over the player's 5
  sorted cards. `pattern = action − 5` (0-31); bit `j` (LSB `j=0` = the
  smallest-indexed sorted card, MSB `j=4` = the largest) is `1` = keep,
  `0` = discard. Discards are replaced with the next cards drawn
  sequentially from `shuffled_deck` at `deck_idx` (already advanced past
  the initial `5 * num_agents` dealt cards). Confirmed extremes: action
  `5` = pattern `00000` = discard all 5 cards; action `36` = pattern
  `11111` = discard none (stand pat). All 32 patterns are legal. Players
  draw in turn starting from the first non-folded player after the small
  blind; once every non-folded player has drawn, the env auto-advances to
  stage 2.

**Observation** (`obs_dim = 54`): indices 0-51 one-hot the current player's
5 hand cards (card id = `suit*13 + (rank-2 or 0 for ace)`, Ace high via
`card % 13`/`card // 13` in `poker_utils.py`); index 52 = current player's
own chips committed this hand; index 53 = max chips committed by any
player this hand.

**Showdown / payout.** `_compare_five_card_hands` scores each live
player's exact 5-card hand (no community cards, no best-of-7) with the
standard 9-category ranking, kicker tie-breaks, and the A-2-3-4-5 "wheel"
straight (`poker_utils.py`). The pot (`sum(chips_in)`) splits equally among
tied winners. If only one player remains active, they take the whole pot
without hand evaluation. Reward per player = chips received at showdown
minus their total pot contribution (zero-sum).

## Simplifications

- **No jokers**, unlike some traditional variants.
- **Single hand per episode.** `reset()` deals exactly one hand; no
  multi-hand match, no dealer-button rotation across hands, no persistent
  bankroll across hands within the environment.
- **Fixed discretized bet sizing, not continuous no-limit.** A raise can
  only be exactly half-pot, exactly pot, or all-in - three fixed sizes
  (plus check/call/fold) - rather than an arbitrary chosen amount.
- **No cap on number of raises** beyond running out of chips (`can_raise`
  only requires `remaining chips > diff`); players can re-raise
  repeatedly until someone is all-in, unlike some fixed-limit games that
  typically cap raises per round.
- **Discard 0-5 cards, no "max 3 unless showing an ace" rule.** All 32
  discard patterns are
  legal every time, including standing pat (action 36, all-ones) and
  discarding the entire hand (action 5, all-zeros).



