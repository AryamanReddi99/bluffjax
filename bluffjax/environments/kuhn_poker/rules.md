# Kuhn Poker — Rules

## Sources

- Kuhn, Harold W. "A simplified two-person poker." *Contributions to the Theory of Games* 1, 1950: 97–103.
- Todd W. Neller and Marc Lanctot, ["An Introduction to Counterfactual Regret Minimization"](https://modelai.gettysburg.edu/2013/cfr/cfr.pdf).


## Objective

Kuhn Poker is the smallest poker variant that still exhibits bluffing at equilibrium. It was introduced by Harold Kuhn in 1950 as a fully tractable extensive-form game (12 information sets) that could be solved by hand, and it has since become the standard toy example for testing game-theoretic algorithms (Nash equilibrium computation, counterfactual regret minimization (CFR), self-play RL, etc.). Because the entire game tree is tiny, an agent's optimal strategy can be derived analytically.

## Setup

- **Deck:** 3 cards of distinct rank — traditionally Jack, Queen, King (ranks 0, 1, 2 in code). Higher rank beats lower rank; there are no suits.
- **Players:** 2.
- **Ante:** Before the deal, each player puts 1 chip into the pot ("antes 1").
- **Deal:** The deck is shuffled and one card is dealt privately to each player; the third card is set aside unseen and plays no further role in the hand.

## Gameplay

- **Turn order:** Play alternates, starting with the first-to-act player (in the canonical description, this is fixed as Player 1).
- **Actions (one per turn):** a player may **pass** (check, if no bet is outstanding; fold, if facing a bet) or **bet** (bet 1 chip, if none is outstanding; call, matching the opponent's 1-chip bet, if facing one). There is a single fixed bet size of 1 chip and no raising.
- **Round end / showdown:** the hand ends as soon as either (a) both players have passed in succession, or (b) both players have bet/called in succession (two matching actions), at which point there is a showdown and the higher card wins the pot; or (c) a player passes after facing a bet (folds), in which case the opponent wins the pot uncontested without a showdown.

| P1 | P2 | P1 | Result | Net payoff |
|----|----|----|--------|-----------|
| pass | pass | - | showdown | +1 to higher card |
| pass | bet | pass | P1 folds | +1 to P2 |
| pass | bet | bet | showdown | +2 to higher card |
| bet | pass | - | P2 folds | +1 to P1 |
| bet | bet | - | showdown | +2 to higher card |

The game is zero-sum in chips: the losing player's payoff is the negative of the winner's.

## Environment Rules

`kuhn_poker.py` implements exactly the game tree above:

- **Cards:** ranks `0,1,2` = Jack/Queen/King. Draw a random permutation of the 3 ranks and deal to the players; the third rank is simply never dealt.
- **Ante / pot:** initialize `pot = [1, 1]` — both players have already anted 1 chip before the first action.
- **Actions:** 2 discrete actions per turn — `0` = check/fold (pass), `1` = bet/call/raise. There is only one bet size (1 chip added to the acting player's `pot` entry), and no re-raising is possible.
- **Showdown winner:** - the player holding the higher-ranked card wins the showdown.
- **Payoffs:** on a terminal step, the winner's reward is net chips won, and the loser's reward is the loss of the chips they put in.
- **Observation:** each player observes their own 3-rank one-hot hand, their own and the opponent's chip count in the pot (1 or 2, one-hot), and a one-hot flag for whether they are the first - or second-to-act player.

## Simplifications vs. commonly-cited variants

Kuhn Poker is already the minimal poker game, so there is very little room for the implementation to diverge from the "textbook" description.
