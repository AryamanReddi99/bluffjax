# Texas Limit Hold'em - Rules

## 1. Sources

- Bowling, Burch, Johanson & Tammelin, "Heads-up Limit Hold'em Poker is Solved," *Science* 347(6218):145-149, 2015 [science.org/doi/abs/10.1126/science.1259433](https://www.science.org/doi/abs/10.1126/science.1259433)

## Objective

Texas Hold'em is a community-card poker game. Each player is dealt two private ("hole") cards and shares five community cards with the table. Players combine their hole cards with the community cards to make the best possible 5-card poker hand. The objective is to win chips, either by having the best hand at showdown or by being the last player remaining after all opponents fold.

## Setup

- **Deck**: a standard 52-card deck (no jokers), shuffled uniformly at random each hand.
- **Players**: real-world Limit Hold'em is played 2-10 handed.
- **Blinds**: before cards are dealt, two players post forced bets - a small blind and a big blind (typically small blind = half the big blind). In heads-up play, the dealer/button posts the small blind and acts first preflop; the other player posts the big blind.
- **Dealing hole cards**: each player receives 2 private hole cards, dealt face down.

## Gameplay

Standard Limit Hold'em is played over **4 betting rounds**:

1. **Preflop** - after hole cards are dealt, betting starts with the player left of the big blind (heads-up: the small blind/button acts first).
2. **Flop** - 3 community cards are dealt face up; a new betting round begins, headed by the first active player left of the button (heads-up: the big blind acts first postflop, since position reverses after the flop).
3. **Turn** - a 4th community card is dealt; another betting round.
4. **River** - a 5th community card is dealt; a final betting round, followed by showdown if 2+ players remain.

**Fixed bet sizes**: in real Limit Hold'em, bets and raises are fixed amounts that depend on the street. Preflop and flop bets/raises equal the big blind (the "small bet"); turn and river bets/raises equal **twice** the big blind (the "big bet"). A "raise" always means bringing one's total contribution for the round up to the current outstanding amount, plus one full bet-sized increment - a player cannot raise by an arbitrary amount, unlike no-limit or pot-limit variants.

**Raise cap**: cardrooms typically cap the number of raises allowed per betting round - commonly described as "bet, raise, re-raise, cap," i.e. an opening bet plus 3-4 further raises, after which players may only call or fold for the rest of that round. Many cardrooms explicitly **suspend this cap in heads-up pots**, allowing unlimited raising once only two players remain, on the reasoning that a cap primarily exists to protect a player from being raised out of a pot by multiple opponents acting in concert - a concern that does not apply one-on-one. The academic HULHE benchmark (Bowling et al., 2015) instead keeps a fixed cap of **4 raises per round** even heads-up, which is the convention this environment follows.

**Action legality**: at any point a player may always fold. A player may check only if no outstanding bet is owed (their contribution already matches the round's current maximum); otherwise they must call, raise (if under the cap), or fold - they cannot check.

**Showdown**: if two or more players reach the river without folding, remaining players reveal their hole cards. Each forms the best possible 5-card hand from their 2 hole cards plus the 5 community cards (best-of-7, i.e. the best hand out of the C(7,5)=21 possible 5-card combinations). The best hand wins the pot; ties split it evenly among the tied winners. If all but one player folds at any point before showdown, that player wins the pot uncontested and hole cards are not required to be shown.

**Hand rankings** (best to worst):

1. Royal flush
2. Straight flush
3. Four of a kind
4. Full house
5. Flush
6. Straight
7. Three of a kind
8. Two pair
9. One pair
10. High card

A royal flush is simply the highest possible straight flush (ten through ace, one suit); most rule sets and evaluators do not treat it as a separate category from "straight flush."

## Environment Rules

| Parameter | Value |
|---|---|
| `small_blind` | 1 |
| `big_blind` | 2 (= 2 × small_blind) |
| `raise_amount` (base, preflop/flop, stages 0-1) | `big_blind` = 2 |
| `raise_amount` (turn/river, stages 2-3) | `2 × raise_amount` = 4 |
| `allowed_raise_num` | 4 (raise cap per betting round) |
| `num_betting_rounds` | 4 (0=preflop, 1=flop, 2=turn, 3=river) |
| `num_agents` (default) | 2 (heads-up) |
| `num_actions` | 4 - 0=call, 1=raise, 2=fold, 3=check |
| `horizon` (default) | 100,000 env steps |
| Deck | 52 cards, encoded as integers 0-51

Turn/action mechanics (`step_env`):
- The raise amount actually charged is doubled once `state.stage >= 2` (turn/river), matching the standard small-bet/big-bet split: `raise_amount = jnp.where(state.stage >= 2, self.raise_amount * 2, self.raise_amount)`.
- `chips_in` tracks each player's **total** chips contributed to the pot across the whole hand (not reset per street), so a `call` always pays `max(chips_in) - chips_in[current_player]`, i.e. exactly enough to match the largest cumulative contributor, and a `raise` pays that same amount plus one `raise_amount` increment.
- `raise_nums[stage]` counts raises taken in the current betting round (indexed 0-3 for preflop/flop/turn/river, reset implicitly since each stage's slot starts at 0); a raise is only legal while `raise_nums[stage] < allowed_raise_num` (4), matching the cap convention. This gives at most 4 raise actions per street.
- A betting round ends once the number of consecutive checks/calls since the last raise (`not_raise_num`) reaches the number of still-active (non-folded) players - i.e. everyone has had a chance to respond to the last raise (or, if nobody raised, everyone has checked/called once).
- The hand ends immediately if only one active player remains (everyone else folded) or once `stage >= 4` (the river betting round has closed, so a showdown occurs).
- Showdown scoring evaluates all `C(7,5) = 21` five-card combinations of each player's 2 hole cards + 5 community cards and takes the maximum-ranked hand, using the standard 10-category ranking (high card through straight flush; royal flush is simply an ace-high straight flush, not scored separately). Ties split the pot evenly among winners.
- Rewards are payoffs normalized by the big blind: `(chips_won - chips_contributed) / big_blind`, i.e. reported in big-blind units (a common convention, cf. mbb/hand, in poker-AI research for comparing strategies independent of stake size).

Observation space (`obs_from_state`, `obs_dim = 72 + num_agents`), documented in the code as following "rlcard limitholdem style":

| Dims | Content |
|---|---|
| 0-51 | One-hot encoding of the acting player's 2 hole cards plus any revealed community cards (flop/turn/river, revealed progressively by `stage`) |
| 52-71 | One-hot encoding of the raise count (0-4) for each of the 4 betting rounds (5 values × 4 rounds = 20 dims) |
| 72-(72+num_agents-1) | One-hot encoding of the acting player's position relative to the small blind (small blind = index 0) |

### Worked example (heads-up, `small_blind=1`, `big_blind=2`)

A concrete hand trace showing the chip mechanics implemented in `step_env`:

1. `reset()` randomly assigns Player A as small blind (posts 1 chip) and Player B as big blind (posts 2 chips). `chips_in = [1, 2]`.
2. Preflop (`stage=0`, `raise_amount=2`), Player A (small blind) acts first:
   - A `raise`s: pays `max(chips_in) - chips_in[A] + 2 = 2 - 1 + 2 = 3`, so `chips_in = [4, 2]`; `raise_nums[0] = 1`.
   - B `call`s: pays `4 - 2 = 2`, so `chips_in = [4, 4]`. Both players have matched with zero pending raises since the last raise (`not_raise_num` reaches the active player count), so the round ends.
3. Flop is dealt (`stage=1`, `raise_amount` still 2). New round starts with the small blind (Player A) acting first in this implementation. Both `check`; round ends, `stage=2`.
4. Turn is dealt (`stage=2`, `raise_amount` doubles to 4). Both `check`; round ends, `stage=3`.
5. River is dealt (`stage=3`, `raise_amount=4`). Both `check`; `stage` becomes 4, so `game_done=True` and a showdown occurs.
6. Payoffs are computed from `_compare_hands` over the best-5-of-7 evaluation; the pot (`sum(chips_in) = 8`) is awarded to the winner (or split on a tie), and each player's reward is `(chips_won - chips_in) / big_blind`.

## Simplifications vs. standard Limit Hold'em

- **No stack/bankroll limit, no all-in, no side pots.** `chips_in` accumulates without bound and a raise always succeeds regardless of any notion of a player's remaining stack - there is no table-stakes constraint, no all-in short-stack handling, and consequently no side-pot logic. This is standard for fixed-limit game-theoretic research environments (bet sizes are small and fixed, so stack depth rarely binds), but is a real divergence from cardroom play where a player can be forced all-in for less than a full bet/raise.
- **Raise cap is fixed at 4 and never waived**, unlike many real cardrooms which suspend the cap once a pot is heads-up. This matches the academic HULHE convention (Bowling et al. 2015) rather than typical cash-room heads-up rules.
- **Postflop first-to-act is always the small blind, not reversed for heads-up.** `step_env` always resets `current_player_idx` to `small_blind_idx` (if still active) at the start of every new betting round, including the flop/turn/river. This is the correct convention for 3+ player Limit Hold'em (the first active player left of the button acts first postflop), but it is **not** correct for the 2-player heads-up special case, where real HULHE rules have the *big blind* act first after the flop (position reverses postflop heads-up). With the default `num_agents=2`, this environment therefore has the small blind acting first in every round of every hand, deviating from standard heads-up play.
