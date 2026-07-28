# Bluff / Cheat / I Doubt It — Rules

This document describes (1) the real-world card game this environment is modeled
after, and (2) the exact game mechanics implemented in `bluff.py`, so that the
differences between the two are explicit.

## References
- Parlett, David. *The A-Z of Card Games*. Oxford Paperback Reference. Oxford: Oxford University Press, 2004. ISBN 978-0-19-860870-7.
- [Cheat / I Doubt It — pagat.com](https://www.pagat.com/beating/cheat.html)


## Objective

Be the first player to get rid of every card in your hand.

## Setup

- A standard 52-card deck (no jokers) is normally used for up to 6 players.
- The deck is shuffled and dealt out as evenly as possible to all players (any remainder cards are added to the central pile).
- Players keep their hands hidden from each other, though hand sizes are visible.

## Gameplay

Play proceeds clockwise in rounds. Each round:

1. **Claim.** The active player announces a rank and a quantity (e.g. "two
   Sevens") and places that many cards face-down on the central pile. In the
   classic variant, the announced rank must be the next rank up from
   the previous round's rank (Ace, 2, 3, ..., King, Ace, ...); players are free
   to lie about whether their face-down cards actually match the announced rank.
2. **Challenge.** Any other player who suspects a lie may call "Cheat!" or 
   "I doubt it!" before the next player plays. The challenged cards are then 
   revealed to everyone.
3. **Resolution.** If the revealed cards do *not* all match the claimed rank,
   the claim was a lie and the player who made it must take the entire pile
   into their hand. If the cards *do* match, the challenge was wrong and the
   challenger takes the entire pile instead. The loser of a challenge starts
   the next round.
4. **No challenge.** If nobody challenges before the next play, the cards stay
   on the pile (uninspected) and play simply continues to the next player /
   next rank.
5. **Game end.** The first player to empty their hand (and not be forced to
   take back the pile as a result of a challenge on that final play) wins.


## Environment Rules

### Default configuration: `num_agents=3`,`num_decks=1`, `num_ranks=13`,`num_suits=4` (a single standard 52-card deck, no jokers, 3 players).

**Setup.** The deck is shuffled. Cards are dealt in equal shares to each of the players.
The leftover cards is placed directly into the
central pile before play starts. A
starting player is chosen uniformly at random.

**Turn cycle.** Each round has three phases:

- **Phase 0 — Claim size.** The active player chooses how many cards to play this turn: 1 to 4, capped both at 4 and at their current total hand size (so a
  player can never claim more than 4 cards, regardless of how many cards they
  hold).
- **Phase 1 — Play cards.** The player then places that many actual cards from
  their hand onto the pile, one at a time. The physical cards played may be of
  any rank(s) actually in the player's hand (bluffing is unrestricted at the
  card level). What rank the play is *declared/claimed* as works as follows:
  - If the previous round ended in a **resolved challenge** (or this is the
    very first round of the game), the claimed rank is free: it is simply
    whatever rank the *first* card the player places happens to be. Every
    other card played in the same turn is then declared as that same rank,
    regardless of its true rank.
  - If the previous round ended with **no challenge**, the claimed rank for
    this round is not a free choice — it is forced by the engine to be exactly
    one rank above the previous round's claimed rank (mod 13, so King wraps
    back to Ace). The player has no agency over the declared rank in this
    case, only over which physical cards they place.
- **Phase 2 — Challenge.** Starting with the player immediately after the one
  who just played, each other player is asked, in fixed clockwise turn order,
  to pass or challenge:
  - The **first** player to challenge immediately resolves the round: the
    played cards (this round only) are compared against the claimed rank and
    count. If they all match, the claim was true and the **challenger** takes
    the entire accumulated pile into their hand; otherwise the claim was false
    and the **player who made the claim** takes the pile. The pile is then
    emptied, and the round's winner becomes the next round's starting player
    with a free choice of rank.
  - If a player passes, the query moves to the next player in turn order.
  - If every other player passes (the query cycles all the way back to the
    original player without a challenge), no challenge occurs: the played
    cards remain on the pile (the pile keeps accumulating across rounds), and
    play passes to the next player after the one who played, whose claimed
    rank is forced to increment as described above.

**Win condition.** A player who plays their last card(s) does **not**
immediately win. The environment only checks for a winner (any player with 0
cards) at the moment a round returns to Phase 0 — i.e., after the challenge
phase concludes. Concretely:
- If nobody challenges the play that empties a player's hand, or someone
  challenges and the claim turns out to be true, that player ends the round
  with 0 cards and the game ends in their favor.
- If someone challenges and the claim was a lie, the player who emptied their
  hand must take the entire pile back — so they no longer have 0 cards, and
  the game continues.

The episode is also truncated after `horizon` environment steps
if no one has won by then.

**Rewards:** the target of
an unchallenged claim is rewarded proportionally to the number of cards
successfully played (`_reward_card`); the winner of a resolved challenge
receives a fixed bonus (`_reward_challenge`) while the loser is penalized in
proportion to the size of the pile they must pick up; and any agent reaching 0
cards at a Phase-0 boundary receives a large terminal bonus (`_reward_win`).

## Simplifications vs. the Standard Game
- **Rank on a fresh claim is whatever card is played first, not a separate
  declaration.** Even when the rank is "free" (after a challenge / at game
  start), there is no explicit "declare a rank" action — this is to simplify learing dynamics.
- **No lying about quantity.** The player always plays exactly as many
  physical cards as they claimed to play; the only bluffing dimension is
  *rank*, not *count* (some real-world variants also allow miscounting).
- **Single challenger resolves the round.** Potential challengers are polled
  once, sequentially, in fixed turn order starting after the player who just
  moved; the first to challenge immediately resolves it.
  Players later in that order never get an independent chance to challenge the
  same play. This differs from the free-for-all nature of real-world
  "anyone can call it out" play.
- **No jokers or wildcards**, and no support for descending/repeating rank
  variants — only the fixed-increment ascending sequence described above.
- **Uneven deal leftover seeds the pile, not a player's hand.** With the
  default 3 players and a 52-card deck, `52 // 3 = 17` cards go to each player
  and the 1 remaining card starts already sitting in the central pile before
  any round is played.
