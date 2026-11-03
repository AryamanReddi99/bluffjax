# Werewolf

## Sources


- [Werewolf Game Rules - Wolfcha](https://www.wolf-cha.com/guides/werewolf-rules)
- [Mafia (party game) - Wikipedia](https://en.wikipedia.org/wiki/Mafia_(party_game))


## Objective

Werewolf is a hidden-role social deduction game played in alternating **night** and **day** rounds by two hidden teams:

- **Villager team** (villagers, doctor, seer): win by identifying and eliminating (via day vote) every werewolf before the werewolves eliminate or outnumber them.
- **Werewolf team**: win by secretly killing villager-team members at night until werewolves equal or outnumber the remaining players.

Players do not know each other's roles at the start except werewolves, who know their fellow werewolves.

## Setup

A standard game seats a mix of ordinary villagers plus a handful of special roles:

- **Werewolves** - the informed minority; they know each other's identities and act as a team at night to kill one player per night.
- **Doctor** (a.k.a. healer) - a villager-team special role who may protect one player per night from the werewolves' kill.
- **Seer** (a.k.a. detective/fortune-teller) - a villager-team special role who may secretly investigate one player per night to learn whether they are a werewolf.
- **Villagers** - ordinary villager-team players with no special night action; they participate only in day discussion/voting.

Each day, all surviving players discuss and vote to eliminate one suspected werewolf.

## Gameplay

**Night phase.** All players "sleep" (are inactive) except those with a role action, who act in turn, each privately:
1. The doctor picks a player to protect from the werewolves' kill that night.
2. The werewolves collectively choose one victim to kill.
3. The seer picks a player to investigate and privately learns whether that player is a werewolf.

If the werewolves' chosen victim matches the doctor's protected player, the
victim survives; otherwise they die and are revealed as dead at the start of
the next day.

**Day phase.** Surviving players discuss openly (accusing, defending,
sharing seer information, etc.), then hold a vote. The player with the most votes is eliminated (executed) and their role is typically revealed. Tied votes are commonly resolved by revote, discussion round, or random tiebreak depending on house rules.

**Win conditions.** The game alternates night -> day -> night -> ... until either:
- All werewolves have been eliminated (**villager team wins**), or
- Werewolves equal or outnumber the remaining players (**werewolf team wins**).

## Environment Rules

**Players and roles.** `num_agents=6` (default), of which `num_werewolves=2`
are werewolves, plus exactly 1 doctor, 1 seer, and 2 villagers. Role are randomly permuted across the 6 seats each episode.

**Actions.** `num_actions = num_agents + 1 = 7`. Actions `0..num_agents-1`
are a **relative target** (`0` = self, `k` = the player `k` seats ahead,
wrapping) converted to an absolute player index via `_to_absolute`; action
`num_agents` (`6`) is **noop** (used for "no target" / pass / abstain, and
is what non-acting players are forced to submit).

**Phases.** `state.phase` is one of three values (matching the obs's phase
one-hot of size 3):

| Phase | Value | Who acts | Actions available |
|---|---|---|---|
| `PHASE_NIGHT` | 0 | one designated actor per sub-phase | target or noop |
| `PHASE_ACCUSE` | 1 | every seat, in order 0..N-1 | target (opt.) or noop |
| `PHASE_VOTE` | 2 | every seat, in order 0..N-1 | target (alive) / noop (dead only) |

Night is itself broken into 4 sequential **sub-phases** (`night_subphase`,
tracked via `night_order = [doctor_idx, seer_idx, werewolf0_idx,
werewolf1_idx]`, fixed at reset):
`NIGHT_DOCTOR(0) -> NIGHT_SEER(1) -> NIGHT_WEREWOLF_0(2) -> NIGHT_WEREWOLF_1(3)`.
Exactly one designated actor takes one action per sub-phase, in that fixed order.

- **Doctor step:** target any living player (including themself -
  `doctor_mask = state.alive` has no self-exclusion) to protect, or noop
  (protects no one).
- **Seer step:** target any living player (self-targeting also allowed) to
  investigate. `seer_results[target]` is immediately set to `+1` if that
  player is a werewolf, `-1` otherwise (villager, doctor, or seer all read
  as `-1`); `0` means "not yet investigated." This array is **never reset
  during the game** (only at `reset()`), so a seer's knowledge accumulates
  and persists across all subsequent nights/days.
- **Werewolf steps:** each living werewolf targets any living
  *non-werewolf* player (`werewolf_mask` excludes werewolf-role players, so
  werewolves can't target each other or themselves) or noops. Stored
  per-werewolf.

After both werewolf sub-phases,
- If the two werewolves agree on a target, that's the kill target.
- If one is dead or nooped, the other's choice is used.
- If they disagree (both alive, both non-noop, different targets), the
  target is chosen uniformly at random between the two.
- If the kill target equals the doctor's protected target, the kill is
  negated (no death) - **the doctor can save themself**.
- The victim (if any) is marked dead.

**Accuse phase.** All `num_agents` seats act **in fixed absolute seat order
0,1,2,...,num_agents-1**, one action each, regardless of alive/dead status.
A living player may target any *other* living player (self-targeting
excluded) as an "accusation," or noop (noop is always legal here, even for
the living) - accusing is optional. A dead player's turn is forced to noop.

**Accusations are purely a broadcast/signaling channel: they do not affect
who gets eliminated.** They are exposed to every player's observation as an
aggregate "was I accused by anyone" bit and are reset only when the
*next* night resolves (so they remain visible through the ensuing vote
phase). After all 6 seats have acted, the phase auto-advances to
`PHASE_VOTE`.

**Vote phase.** Again all `num_agents` seats act in fixed absolute order
0..num_agents-1. A living player must vote for a living player other than
themself (**noop is not a legal action for living players in this phase** -
`vote_avail`'s noop slot is only enabled when `~alive[cp]`, i.e. only dead
seats may/must noop). `votes[seat]` records the choice.

After all 6 seats have acted, game tallies votes from
living voters only, breaks ties randomly and eliminates a player. If the game isn't
over, the phase resets to `PHASE_NIGHT`, night buffers (`doctor_target`,
`seer_target`, `werewolf_targets`, `votes`) are cleared, and play continues
with the doctor.

**Episode length / timeout.** `horizon=200` individual action-steps (not
rounds - one full night+accuse+vote round consumes `4 + 6 + 6 = 16` steps,
so roughly 12 rounds fit in the default horizon). If `timestep >= horizon`
before either team wins, the episode is forcibly ended with `game_winner=0`,
i.e. **the villager team is awarded the win on timeout**.

**Rewards.** All rewards are `0` on non-terminal steps. On the terminal
step, every agent whose role's team matches `game_winner` gets `+10`;
everyone else gets `-10`.

**Observation** (`obs_dim = 4 + 3 + num_agents + num_agents + num_agents*3 + num_agents = 43` for default `num_agents=6`), all rolled to the acting player's relative perspective (self = index 0):
1. Own role, one-hot (4).
2. Current phase, one-hot (3).
3. Alive mask, relative (`num_agents`).
4. Werewolf teammates mask, relative - all zero unless the observer is a werewolf, in which case it flags the other living werewolf (`num_agents`).
5. Seer results, ternary one-hot per seat (`num_agents * 3`, encoding `{-1,0,1} -> 3`-dim one-hot) - all zero unless the observer is the seer.
6. "Was I accused by anyone this round" bit per seat, relative (`num_agents`).

## Simplifications

- **No natural-language discussion.** The social mechanics of default Werewolf/Mafia - free-form verbal argument, persuasion, lying, and
  claiming roles - are not implemented here. The inter-player
  communication is the discrete accuse action, broadcast only as an aggregate "accused by someone" bit and the vote action itself.
- **Vote ties are broken by uniform random noise**, not by a revote, PK
  (player-kill) speech round, or "no elimination" rule as in some rulesets
  (e.g. Wolfcha's tied-players revote).
- **Fixed, minimal role set.** Only villager/werewolf/doctor/seer exist - no hunter, witch, bodyguard, cupid,
  minion, tanner, or other special roles - these might be impplemented in future versions!.
- **Reward structure is a simple terminal team-outcome signal** (`+10`/`-10`
  split strictly along werewolf-vs-not role lines), with no shaping for
  individual actions (e.g., no bonus for a correct accusation or a
  successful save), and a 200-step hard timeout that defaults to a
  villager-team win.
