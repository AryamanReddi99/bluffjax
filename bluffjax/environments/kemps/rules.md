# Kemps

## Sources

- Parlett, David. *The A-Z of Card Games*. Oxford Paperback Reference. Oxford: Oxford University Press, 2004. ISBN 978-0-19-860870-7.
- Pagat.com, ["Rules of Card Games: Kemps"](https://www.pagat.com/commerce/kemps.html)
- Bicycle Cards, ["Kemps"](https://bicyclecards.com/how-to-play/kemps/)


## Objective

Each player is partnered with the player seated across the table.
A team wins a hand by either (a) collecting four cards of one rank and
covertly signaling this to their partner, who then calls out "KEMPS", or
(b) noticing an opposing team has four-of-a-kind and calling "STOP KEMPS"
first.

## Setup

- Usually 4 players (scales up to 4-12 players / 2-6 teams of two in principle).
- A standard 52-card deck (a second deck for larger groups in some
  variants).
- 4 cards face-down to each player and 4 cards face-up in a
  row in the center.

## 4. Gameplay

Play is simultaneous and unstructured. Any player may pick
up one of the four face-up center cards and
immediately discard a card from their hand, always keeping
exactly 4 cards. Once nobody
wants the exposed center cards, they are discarded for the round and four fresh cards are dealt.

Once a player holds four-of-a-kind, they try to covertly flash their
signal to their partner without the opposing team noticing. The partner,
recognizing it, calls "KEMPS!", revealing their hand: if it really is
four-of-a-kind they win the round; if wrong, the calling team wins
instead. Any player may instead call "CAUGHT!" on suspicion that an
opponent has four-of-a-kind; the accused reveals their hand, and whichever
side was wrong loses. The hand ends the instant a call is
resolved. Some variants count to a specific number of rounds per game.

## Environment Rules

### Default configuration: `num_agents=4`, `num_ranks=13`, `num_suits=4`, `hand_size=4`, `comm_dim=2`, `horizon=200`.

**Teams/partners** Partner = `(agent_idx + 2) % num_agents`. At the default `num_agents=4` this is: team 0 = {0,2}, team 1 = {1,3}.

**Dealing** Deck is shuffled; private cards dealt
`hand_size` each, next 4 form the center, rest form a face-down stock drawn as the center refreshes. `state.communication` (each agent's last signal) starts all-zero.

**Action encoding** (`step_env`). Each agent's action integer decomposes as
`game_action * comm_dim + comm_signal`:
- `game_action < num_ranks^2`: **swap** - `lose_rank = game_action //
  num_ranks`, `gain_rank = game_action % num_ranks`; discard one card of
  `lose_rank` from hand for one card of `gain_rank` from the center
  (`lose_rank == gain_rank` is a legal, pointless, degenerate swap).
- `game_action == num_ranks^2`: **NOOP**.
- `game_action == num_ranks^2 + 1`: **KEMPS** - claim partner has
  four-of-a-kind.
- `game_action == num_ranks^2 + 2`: **CAUGHT** - accuse the opposing
  team of four-of-a-kind.
- `comm_signal in [0, comm_dim)` rides along with *every* action, is not a separate action type, and no "no
  signal" option exists.

**Resolving a step.** If nobody declares and not everyone chose NOOP,
swaps are applied one agent at a time in a random order. If *every* agent chose NOOP, the center is
swept and 4 new cards dealt; if the stock can't supply
4 more, the hand ends in a scoreless draw (`real_deal`). If any agent
declared KEMPS and/or CAUGHT, no swap happens that step -
declarations are resolved instead and the hand ends.

**Declaration resolution.** If multiple calls are declared in the same step, whichever type the lowest-indexed declaring agent used is the one that resolves (if only one type was declared, that type resolves regardless of index); among agents declaring the winning type, the lowest-indexed one is "the" caller.
- **KEMPS**: checks if the caller's partner
truly holds four-of-a-kind. If true, caller's team gets
  `+1`, other team `-1`; if false, reversed.
- **CAUGHT**: checks if *any* member of the opposing
  team currently holds four-of-a-kind. If true, accusing team `+1`, accused
  team `-1`; if false, reversed.

All other steps give reward 0 to everyone. A hand is one episode: episode ends on any resolved declaration, the stock-exhausted draw, or
`timestep+1 >= horizon`.

**Observations** (`obs_from_state`). Concatenation of: (1) an
`(num_agents, deck_size)` block, all-zero except the observing agent's own
row (its own hand, binary) — hands are private; (2) the `deck_size` binary
indicator of the 4 center cards, identical for all agents (public); (3)
every agent's last one-hot public signal, rolled relative to the observer's own
index so index 0 is always "my last signal".

## Simplifications vs. the standard real-world game

- **The signal channel is public and explicit, not a hidden gesture.** In
  the real game, catching a signal is a perception problem - a covert,
  deniable gesture opponents must actually notice. Here every agent's
  `comm_signal` from the previous step appears in *every* agent's
  observation (relative-rolled); the raw value is never hidden from
  opponents, so there is no "the other team failed to notice" outcome.
  Only its *meaning* is left for policies to learn - bluffing/concealment
  can only emerge from how a fully-visible value gets used, not from
  opponents literally failing to perceive it.
- **A signal rides on every action, every step**, regardless of whether the
  agent actually has four-of-a-kind, rather than the real game's free
  signalling.
- **Swap moves exactly one card at a time**; some real-world descriptions
  allow taking multiple center cards in one move.
- **"First to touch it" is replaced by a random per-step agent order**
