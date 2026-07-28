# Seven Card Stud - Rules

## Sources

- [Seven Card Stud - poker rules, pagat.com](https://www.pagat.com/poker/variants/7stud.html)
- [Seven-Card Stud Rules - PokerNews](https://www.pokernews.com/poker-rules/seven-card-stud.htm)


## Objective
 
Seven Card Stud is a fixed-limit, non-community-card poker game. Each player is dealt seven private cards over five betting rounds (some face down, some face up). At showdown, each player forms the best possible 5-card poker hand out of their 7 cards; the best hand (or last player remaining after all others fold) wins the pot.

## Setup

- **Deck**: standard 52-card deck, no community cards, no burn cards.
- **Players**: no fixed limit in the standard game, though heads-up (2 players) and short-handed tables are common; a maximum of 7 players can be dealt a full 7 cards from a single 52-card deck.
- **Ante**: every player posts a small mandatory ante before any cards are dealt.
- **Initial deal ("3rd street")**: each player receives two cards face down (hole cards) and one card face up (the "door card").

## Gameplay

### Bring-in (3rd street)
After the initial deal, the player showing the lowest-ranked door card must post a forced "bring-in" bet, which is smaller than a full small bet. Ties among lowest cards are broken by suit rank, low to high: clubs, diamonds, hearts, spades. The bring-in player may post the minimum bring-in or "complete" (bet the full small bet); the next player, acting clockwise, may then fold, call, or raise.

### Streets and dealing
- **3rd street**: 2 down + 1 up, bring-in, first betting round.
- **4th street**: one more up card dealt to each remaining player; betting round.
- **5th street**: one more up card dealt; betting round (big bet now applies).
- **6th street**: one more up card dealt; betting round.
- **7th street ("the river")**: one final card dealt face down to each remaining player; final betting round; showdown follows.

By 7th street each surviving player has 7 cards: 3 hidden (two from 3rd street, one from 7th street) and 4 exposed up cards.

### Action order
On 3rd street, action starts with the bring-in player's forced post, and betting then proceeds clockwise starting from the next player. On every subsequent street (4th-7th), the player whose exposed up cards make the best poker hand acts first (evaluated only on the visible up cards - pairs, trips, etc. count, but incomplete straights/flushes do not); action then proceeds clockwise. Ties in the best-shown hand are broken by suit, using the same low-to-high club/diamond/heart/spade order used for the bring-in.

### Small bet vs. big bet
Fixed-limit stud uses two bet sizes: a small bet on 3rd and 4th street, and a big bet (double the small bet) from 5th street onward.

### Raise cap
Betting is capped at one bet plus a fixed number of raises per round (commonly three or four, house-dependent), often uncapped once only two players remain.

### Showdown and hand ranking
Remaining players reveal their two hidden hole cards (7th street's down card stays private if the player folded or wins uncontested but is shown at a called showdown) and form the best 5-card hand out of their 7 cards, using standard poker hand rankings, best to worst:

1. Straight flush
2. Four of a kind
3. Full house
4. Flush
5. Straight
6. Three of a kind
7. Two pair
8. One pair
9. High card

Ties at showdown split the pot evenly.

## Environment Rules

- **Ante**: `ante=0.5`. In `reset()`, every player's `chips_in` is initialized to `ante` before anything else.
- **Bring-in**: `bring_in=0.5`. `reset()` deals all 7 cards to each player at once (`shuffled_deck[:7*num_agents]`), takes each player's door card (index 2, the first up card), and finds the lowest via `_get_bring_in_idx` (rank 2 low, Ace high; ties broken exactly by suit order clubs < diamonds < hearts < spades, matching real stud). That player's `chips_in` gets `+= bring_in` on top of the ante (so the bring-in player's total posted is `ante + bring_in = 1.0`).
- the amount is applied automatically in `reset()`, and the first actual *action* (call/raise/fold/check) belongs to the next non-folded player clockwise.
- **Streets / stages**: `state.stage` runs 0-4, representing 3rd through 7th street (`num_betting_rounds=5`). Up-card visibility to opponents is controlled by `NUM_VISIBLE_BY_STAGE = [1, 2, 3, 4, 4]` applied to `agent_cards[:, 2:6]`: stage 0 shows 1 up card (the door card), stage 1 shows 2, stage 2 shows 3, stages 3 and 4 show all 4. Card index 6 (the 7th-street card) is never exposed in the observation - it functions as the final down card, matching real stud's "2 down, 4 up, 1 down" pattern. All 7 cards are actually dealt up front in `reset()` for JAX-vectorization efficiency; only their *visibility* in the observation is staged street by street.
- **Small/big bet split**: `small_bet=1`, `big_bet=2`. In `step_env()`, `raise_amount = big_bet` when `stage >= 2` (5th, 6th, 7th street) and `small_bet` otherwise (3rd, 4th street) - exactly the standard split. There is no "big bet kicks in early if a pair is showing on 4th street" exception; the switch is purely stage-based.
- **Raise cap**: `allowed_raise_num=4`. `get_avail_actions()` disables the raise action once `raise_nums[stage] >= 4`, i.e. up to 4 raises (5 total bets) per betting round, applied uniformly regardless of how many players remain (no heads-up unlimited-raise exception).
- **First-to-act on later streets**: `_get_first_to_act_by_upcards()` scores each non-folded player's exposed up cards (2 cards for stage 1, 3 for stage 2, 4 for stages 3-4) using `_score_visible_upcards` (high card/pair/two pair/trips/quads over the partial up-card set) and picks the `argmax`.
- **Betting/action set**: 4 actions - `0=call, 1=raise, 2=fold, 3=check` - identical in structure to the repo's Texas Hold'em Limit environment.
- **Round/hand end**: a betting round ends when `not_raise_num >= num_active_players` (everyone has acted since the last raise); the hand ends when only one player remains unfolded or `stage` reaches 5 (after 7th-street betting completes).
- **Showdown**: `_compare_hands` scores each non-folded player's full 7 cards over all `C(7,5)=21` five-card combinations (`_score_seven_card_hand`) and awards the pot to the max-score player(s), split evenly on ties.
- **Payoffs**: rewards are `(chips won - chips contributed) / big_bet`, i.e. normalized in big-bet units.
- **Episodic structure**: `reset()` deals a brand-new hand from a fresh shuffle every time; `step()` auto-resets when `done`. There is no persistent bankroll across hands - each hand starts from the same `ante`/`bring_in` postings with no memory of prior stack depletion.
- **Horizon**: `horizon=100_000` (env-level cap on total timesteps, not specific to stud rules).

### Worked example (heads-up, default params)

With `num_agents=2`, `ante=0.5`, `bring_in=0.5`, `small_bet=1`, `big_bet=2`:

1. `reset()`: both players post `ante=0.5` (`chips_in = [0.5, 0.5]`). The player with the lower door card additionally posts `bring_in=0.5`, so that player's `chips_in` becomes `1.0` while the other's stays `0.5`.
2. The other player (first to act) faces `max_chips=1.0` with only `0.5` in - they cannot check (`can_check` requires `player_chips == max_chips`). They can call (match to `1.0`), raise (match to `1.0` then add `small_bet=1` more, i.e. to `2.0`, since `stage=0 < 2`), or fold.
3. Once 3rd-street betting is settled (`not_raise_num >= num_active_players`), `stage` becomes 1 (4th street), a new up card is exposed to both players in the observation, and `_get_first_to_act_by_upcards` determines who acts first based on the best 2-card up-card hand.
4. This repeats through `stage=2,3` (5th, 6th street), where `raise_amount` switches to `big_bet=2` because `stage >= 2`.
5. At `stage=4` (7th street) after the final betting round, `new_stage >= 5` triggers `game_done`, `_compare_hands` evaluates each player's best 5-of-7 hand, and the pot (`sum(chips_in)`) is split among winners.

## 6. Simplifications

- **No persistent bankroll across hands.** Each call to `reset()` starts a fresh hand with the same fixed ante/bring-in structure; chip stacks are not carried over between hands, so there is no elimination when a player's stack is depleted (unlike a cash game or tournament).
- **Fixed raise cap of 4 regardless of player count.** Real cardrooms typically cap raises at 3 (bet + 3 raises) with more than 2 active players, and sometimes remove the cap entirely heads-up; the environment always caps at 4 raises per round no matter how many players remain active.
- **No "big bet trigger" exception on 4th street.** Standard cardroom rules often let players opt into the big bet on 4th street if a pair is showing; the environment always uses the small bet on stages 0-1 and the big bet on stages 2-4, with no exception.
- **First-to-act on 4th-7th street IS exposed-card-based, matching real stud** - this is not simplified away. `_get_first_to_act_by_upcards` genuinely computes the best partial hand from each player's visible up cards and lets that player act first, as in casino rules, rather than using a fixed rotation.
- **No suit tiebreak for "best hand showing."** Real stud breaks ties in the exposed-hand ranking by suit (clubs low, spades high); this environment's `_get_first_to_act_by_upcards` breaks such ties arbitrarily by player index.
