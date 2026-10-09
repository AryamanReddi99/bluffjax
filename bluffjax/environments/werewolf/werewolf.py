"""
Werewolf (Mafia) social deduction environment.

Roles: num_werewolves werewolves, one doctor, one seer and villagers for the
rest, dealt uniformly at random to the player indices. The doctor, the seer
and the villagers are the humans.

A game cycles night -> accuse -> vote until a team has won:
- Night: the doctor protects a living player (itself included), the seer
  learns whether another living player is a werewolf, and each werewolf picks
  a living human to attack. The victim is the werewolves' most picked target
  (ties broken uniformly at random) and dies unless the doctor protected it.
- Accuse: each living player in turn accuses another living player or passes.
  Everyone sees which players have been accused so far that day.
- Vote: each living player in turn votes for another living player. The most
  voted player (ties broken uniformly at random) is eliminated.

The humans win as soon as no werewolf is alive and the werewolves as soon as
they are at least as many as the living humans. Winners get +10 and losers
-10. A game that reaches the horizon ends with no winner and no reward.

Turn order: the player index is only an internal ordering. The first speaker
of the first day is drawn at reset and moves on to the next living player
every day; the accuse and vote turns go round the living players starting
from the day's first speaker. At night the doctor acts first, then the seer,
then the werewolves in an order drawn at reset. Dead players never get a turn.

Observations and actions are relative to the player to act: relative index j
is the player j seats after it (0 is itself), and action n is the no-op.
"""

import jax
import jax.numpy as jnp
from jax import lax
from flax import struct
from functools import partial
from bluffjax.utils.typing import (
    Any,
    FloatArray,
    IntArray,
    BoolArray,
    PRNGKeyArray,
)
from bluffjax.environments.env import AECEnv
from bluffjax.environments.spaces import Discrete

# Role constants
VILLAGER = 0
WEREWOLF = 1
DOCTOR = 2
SEER = 3

# Phase constants
PHASE_NIGHT = 0
PHASE_ACCUSE = 1
PHASE_VOTE = 2

# Night slots (index into night_order): doctor, seer, then the werewolves
NIGHT_DOCTOR = 0
NIGHT_SEER = 1
NIGHT_WEREWOLF = 2

WIN_REWARD = 10.0


@struct.dataclass
class WerewolfState:
    """Werewolf game state. Player references are absolute indices, -1 = none."""

    roles: IntArray  # (num_agents,) 0=villager, 1=werewolf, 2=doctor, 3=seer
    alive: BoolArray  # (num_agents,)
    phase: IntArray  # 0=night, 1=accuse, 2=vote
    night_subphase: IntArray  # night slot of the player to act (see night_order)
    current_player_idx: IntArray
    # First speaker of the current day; during the night, the player from
    # which the next day's first speaker is searched (first living one).
    start_player_idx: IntArray

    # Night order: [doctor, seer, werewolf_0, ..., werewolf_{k-1}]
    night_order: IntArray  # (2 + num_werewolves,)

    # Tonight's choices
    doctor_target: IntArray
    seer_target: IntArray
    werewolf_targets: IntArray  # (num_werewolves,) pick of night_order[2 + i]

    # Seer's investigation results: (num_agents,) -1=human, 0=unchecked, 1=werewolf
    seer_results: FloatArray

    # Accuse phase: (num_agents,) who each player accused today
    accusations: IntArray

    # Vote phase: (num_agents,) who each player voted for today
    votes: IntArray

    # Turns taken in the current accuse or vote phase
    phase_progress: IntArray

    absorbing: BoolArray
    done: BoolArray
    game_winner: IntArray  # -1=none (yet, or timeout), 0=humans, 1=werewolves
    timestep: IntArray


class Werewolf(AECEnv):
    """Werewolf AEC environment with doctor and seer roles."""

    def __init__(
        self,
        num_agents: int = 6,
        num_werewolves: int = 2,
        horizon: int = 200,
    ) -> None:
        super().__init__(num_agents=num_agents, horizon=horizon)
        if num_werewolves < 1:
            raise ValueError(f"num_werewolves must be >= 1, got {num_werewolves}")
        if 2 * num_werewolves >= num_agents:
            raise ValueError(
                f"{num_werewolves} werewolves among {num_agents} players would "
                "win before the first night; the humans must outnumber them"
            )
        self.num_werewolves = num_werewolves
        self.num_roles = 4  # villager, werewolf, doctor, seer
        self.num_night_actors = 2 + num_werewolves  # doctor, seer, werewolves

        # Role of the player at night_order[i] for i < num_night_actors
        self.role_template = jnp.array(
            [DOCTOR, SEER]
            + [WEREWOLF] * num_werewolves
            + [VILLAGER] * (num_agents - 2 - num_werewolves),
            dtype=jnp.int32,
        )

        # Action: 0 to num_agents-1 = relative target, num_agents = noop
        self.num_actions = num_agents + 1

        # Obs dim: role(4) + phase(3) + alive(num_agents) + werewolf_teammates(num_agents)
        # + seer_results(num_agents*3 for ternary) + accusations(num_agents)
        self.obs_dim = 4 + 3 + num_agents + num_agents + num_agents * 3 + num_agents

    def _seats_from(self, start: IntArray) -> IntArray:
        """Absolute indices start, start + 1, ..., start + n - 1 (mod n)."""
        return (start + self.agent_idxs) % self.num_agents

    def _roll_for_perspective(
        self, arr: FloatArray, current_player: IntArray
    ) -> FloatArray:
        """Roll array so current player is at index 0 (relative perspective)."""
        return jnp.roll(arr, -current_player, axis=0)

    def _first_alive_from(self, start: IntArray, alive: BoolArray) -> IntArray:
        """First living player at or after start, going round the table."""
        seats = self._seats_from(start)
        return seats[jnp.argmax(alive[seats])]

    def _next_night_slot(
        self, slot: IntArray, alive: BoolArray, night_order: IntArray
    ) -> IntArray:
        """First night slot after slot whose player is alive, or
        num_night_actors if there is none."""
        slots = jnp.arange(self.num_night_actors)
        candidate = (slots > slot) & alive[night_order]
        return jnp.where(
            candidate.any(), jnp.argmax(candidate), self.num_night_actors
        ).astype(jnp.int32)

    def _most_chosen(
        self, counts: FloatArray, rng: PRNGKeyArray, start: IntArray
    ) -> IntArray:
        """Player with the most choices, ties broken uniformly at random
        (-1 if nobody was chosen). The tie-break draws are taken in seat order
        from start, so they don't favour any fixed index."""
        noise = jnp.roll(jax.random.uniform(rng, (self.num_agents,)), start)
        choice = jnp.argmax(counts + 0.5 * noise)
        return jnp.where(counts.max() > 0, choice, -1).astype(jnp.int32)

    def _winner(self, alive: BoolArray, roles: IntArray) -> IntArray:
        """0 if the humans have won, 1 if the werewolves have, -1 otherwise."""
        num_ww = jnp.sum(alive & (roles == WEREWOLF))
        num_humans = jnp.sum(alive & (roles != WEREWOLF))
        winner = jnp.where(num_ww >= num_humans, 1, -1)
        return jnp.where(num_ww == 0, 0, winner).astype(jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def obs_from_state(self, state: WerewolfState) -> FloatArray:
        """Observation for current player, all relative."""
        cp = state.current_player_idx

        # 1. One-hot role (4 dims)
        role_oh = jax.nn.one_hot(state.roles[cp], self.num_roles, dtype=jnp.float32)

        # 2. One-hot phase (3 dims)
        phase_oh = jax.nn.one_hot(state.phase, 3, dtype=jnp.float32)

        # 3. Binary living players, relative (num_agents dims)
        alive_relative = self._roll_for_perspective(state.alive.astype(jnp.float32), cp)

        # 4. Werewolf teammates relative (num_agents dims) - only non-zero if current is werewolf
        is_werewolf = state.roles[cp] == WEREWOLF
        teammate_mask = (state.roles == WEREWOLF) & (self.agent_idxs != cp)
        teammate_relative = self._roll_for_perspective(
            teammate_mask.astype(jnp.float32), cp
        )
        werewolf_teammates = jnp.where(
            is_werewolf, teammate_relative, jnp.zeros(self.num_agents)
        )

        # 5. Seer results: ternary per player (-1, 0, 1), relative. Encode as 3 values each.
        seer_rel = self._roll_for_perspective(state.seer_results, cp)
        is_seer = state.roles[cp] == SEER
        seer_encoded = jax.nn.one_hot(
            (seer_rel + 1).astype(jnp.int32), 3, dtype=jnp.float32
        ).reshape(-1)
        seer_obs = jnp.where(is_seer, seer_encoded, jnp.zeros(self.num_agents * 3))

        # 6. Accusations: binary relative - who has been accused today
        accused_by_any = jnp.any(
            state.accusations[None, :] == self.agent_idxs[:, None], axis=1
        ).astype(jnp.float32)
        accusations_relative = self._roll_for_perspective(accused_by_any, cp)

        obs = jnp.concatenate(
            [
                role_oh,
                phase_oh,
                alive_relative,
                werewolf_teammates,
                seer_obs,
                accusations_relative,
            ]
        )
        return obs.astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: WerewolfState) -> BoolArray:
        """Available actions of the player to act. Action i targets the player
        i seats after it (relative index i); action num_agents is the no-op.

        Night: the doctor may protect any living player (itself included), the
        seer may investigate any other living player and a werewolf may attack
        any living human. Accuse: any other living player, or pass (no-op).
        Vote: any other living player. The no-op is also available when no
        target is (never the case in a reachable state)."""
        cp = state.current_player_idx
        seats = self._seats_from(cp)  # absolute index of each relative target
        alive = state.alive[seats]
        not_self = self.agent_idxs != 0
        slot = state.night_subphase
        night_targets = jnp.where(
            slot == NIGHT_DOCTOR,
            alive,
            jnp.where(
                slot == NIGHT_SEER,
                alive & not_self,
                alive & (state.roles[seats] != WEREWOLF),
            ),
        )
        day_targets = alive & not_self
        targets = jnp.where(state.phase == PHASE_NIGHT, night_targets, day_targets)
        noop = (state.phase == PHASE_ACCUSE) | ~targets.any()
        return jnp.concatenate([targets, noop[None]])

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: PRNGKeyArray) -> tuple[WerewolfState, FloatArray]:
        """Initialize game with random role assignment and first speaker."""
        rng_roles, rng_start = jax.random.split(rng)

        # The player at seats[i] gets role_template[i]: the first two are the
        # doctor and the seer, the next num_werewolves the werewolves in their
        # (random) night order.
        seats = jax.random.permutation(rng_roles, self.num_agents).astype(jnp.int32)
        roles = (
            jnp.zeros(self.num_agents, dtype=jnp.int32)
            .at[seats]
            .set(self.role_template)
        )
        night_order = seats[: self.num_night_actors]
        start_player = jax.random.randint(rng_start, (), 0, self.num_agents).astype(
            jnp.int32
        )

        state = WerewolfState(
            roles=roles,
            alive=jnp.ones(self.num_agents, dtype=bool),
            phase=jnp.int32(PHASE_NIGHT),
            night_subphase=jnp.int32(NIGHT_DOCTOR),
            current_player_idx=night_order[NIGHT_DOCTOR],
            start_player_idx=start_player,
            night_order=night_order,
            doctor_target=jnp.int32(-1),
            seer_target=jnp.int32(-1),
            werewolf_targets=jnp.full(self.num_werewolves, -1, dtype=jnp.int32),
            seer_results=jnp.zeros(self.num_agents, dtype=jnp.float32),
            accusations=jnp.full(self.num_agents, -1, dtype=jnp.int32),
            votes=jnp.full(self.num_agents, -1, dtype=jnp.int32),
            phase_progress=jnp.int32(0),
            absorbing=jnp.zeros(self.num_agents, dtype=bool),
            done=jnp.bool_(False),
            game_winner=jnp.int32(-1),
            timestep=jnp.int32(0),
        )
        obs = self.obs_from_state(state)
        return state, obs

    def _end_or(self, state: WerewolfState, continue_fn) -> WerewolfState:
        """Ends the game if a team has won, otherwise applies continue_fn."""
        winner = self._winner(state.alive, state.roles)
        return lax.cond(
            winner >= 0,
            lambda s: s.replace(
                done=jnp.bool_(True),
                absorbing=jnp.ones(self.num_agents, dtype=bool),
                game_winner=winner,
            ),
            continue_fn,
            state,
        )

    def _start_day(self, state: WerewolfState) -> WerewolfState:
        """Accuse phase from the first living player at or after start_player_idx."""
        first = self._first_alive_from(state.start_player_idx, state.alive)
        return state.replace(
            phase=jnp.int32(PHASE_ACCUSE),
            start_player_idx=first,
            current_player_idx=first,
            accusations=jnp.full(self.num_agents, -1, dtype=jnp.int32),
            phase_progress=jnp.int32(0),
        )

    def _start_vote(self, state: WerewolfState) -> WerewolfState:
        """Vote phase from the day's first speaker."""
        return state.replace(
            phase=jnp.int32(PHASE_VOTE),
            current_player_idx=state.start_player_idx,
            votes=jnp.full(self.num_agents, -1, dtype=jnp.int32),
            phase_progress=jnp.int32(0),
        )

    def _start_night(self, state: WerewolfState) -> WerewolfState:
        """Night from its first living actor; the next day's first speaker
        moves on to the next living player."""
        slot = self._next_night_slot(jnp.int32(-1), state.alive, state.night_order)
        return state.replace(
            phase=jnp.int32(PHASE_NIGHT),
            night_subphase=slot,
            current_player_idx=state.night_order[slot],
            start_player_idx=self._first_alive_from(
                state.start_player_idx + 1, state.alive
            ),
            doctor_target=jnp.int32(-1),
            seer_target=jnp.int32(-1),
            werewolf_targets=jnp.full(self.num_werewolves, -1, dtype=jnp.int32),
            phase_progress=jnp.int32(0),
        )

    def _resolve_night(self, state: WerewolfState, rng: PRNGKeyArray) -> WerewolfState:
        """Resolve night: apply the attack unless healed, then check for a win."""
        werewolves = state.night_order[NIGHT_WEREWOLF:]
        picks = jnp.where(state.alive[werewolves], state.werewolf_targets, -1)
        counts = jnp.sum(
            picks[:, None] == self.agent_idxs[None, :], axis=0, dtype=jnp.float32
        )
        victim = self._most_chosen(counts, rng, state.start_player_idx)
        healed = victim == state.doctor_target
        killed = (self.agent_idxs == victim) & ~healed
        state = state.replace(alive=state.alive & ~killed)
        return self._end_or(state, self._start_day)

    def _resolve_vote(self, state: WerewolfState, rng: PRNGKeyArray) -> WerewolfState:
        """Resolve vote: eliminate the most voted player, then check for a win."""
        valid = state.alive[:, None] & (
            state.votes[:, None] == self.agent_idxs[None, :]
        )
        counts = jnp.sum(valid, axis=0, dtype=jnp.float32)
        eliminated = self._most_chosen(counts, rng, state.start_player_idx)
        state = state.replace(alive=state.alive & (self.agent_idxs != eliminated))
        return self._end_or(state, self._start_night)

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self, rng: PRNGKeyArray, state: WerewolfState, action: IntArray
    ) -> tuple[
        WerewolfState,
        FloatArray,
        FloatArray,
        BoolArray,
        bool,
        dict[str, Any],
    ]:
        """Execute one step."""
        cp = state.current_player_idx
        # Relative action -> absolute target (-1 for the no-op)
        target = jnp.where(
            action < self.num_agents, (cp + action) % self.num_agents, -1
        ).astype(jnp.int32)

        def do_night(s: WerewolfState) -> WerewolfState:
            slot = s.night_subphase
            is_seer = slot == NIGHT_SEER
            checked = is_seer & (self.agent_idxs == target)
            seer_results = jnp.where(
                checked,
                jnp.where(s.roles == WEREWOLF, 1.0, -1.0),
                s.seer_results,
            )
            werewolf_slot = jnp.arange(self.num_werewolves) == slot - NIGHT_WEREWOLF
            s = s.replace(
                doctor_target=jnp.where(slot == NIGHT_DOCTOR, target, s.doctor_target),
                seer_target=jnp.where(is_seer, target, s.seer_target),
                seer_results=seer_results,
                werewolf_targets=jnp.where(werewolf_slot, target, s.werewolf_targets),
            )
            next_slot = self._next_night_slot(slot, s.alive, s.night_order)
            return lax.cond(
                next_slot < self.num_night_actors,
                lambda s_: s_.replace(
                    night_subphase=next_slot,
                    current_player_idx=s_.night_order[
                        jnp.minimum(next_slot, self.num_night_actors - 1)
                    ],
                ),
                lambda s_: self._resolve_night(s_, rng),
                s,
            )

        def next_speaker(s: WerewolfState) -> WerewolfState:
            return s.replace(
                current_player_idx=self._first_alive_from(
                    s.current_player_idx + 1, s.alive
                )
            )

        def do_accuse(s: WerewolfState) -> WerewolfState:
            s = s.replace(
                accusations=s.accusations.at[cp].set(target),
                phase_progress=s.phase_progress + 1,
            )
            return lax.cond(
                s.phase_progress >= jnp.sum(s.alive),
                self._start_vote,
                next_speaker,
                s,
            )

        def do_vote(s: WerewolfState) -> WerewolfState:
            s = s.replace(
                votes=s.votes.at[cp].set(target),
                phase_progress=s.phase_progress + 1,
            )
            return lax.cond(
                s.phase_progress >= jnp.sum(s.alive),
                lambda s_: self._resolve_vote(s_, rng),
                next_speaker,
                s,
            )

        next_state = lax.switch(state.phase, [do_night, do_accuse, do_vote], state)

        # Rewards when a team wins at this step
        decided = (next_state.game_winner >= 0) & (state.game_winner < 0)
        team_won = jnp.where(
            state.roles == WEREWOLF,
            next_state.game_winner == 1,
            next_state.game_winner == 0,
        )
        rewards = jnp.where(
            decided, jnp.where(team_won, WIN_REWARD, -WIN_REWARD), 0.0
        ).astype(jnp.float32)

        # Horizon: the game ends with no winner and no reward
        next_timestep = state.timestep + 1
        done = next_state.done | (next_timestep >= self.horizon)
        next_state = next_state.replace(
            timestep=next_timestep,
            done=done,
            absorbing=jnp.broadcast_to(done, (self.num_agents,)),
        )

        obs = self.obs_from_state(next_state)
        player_won = jnp.where(
            next_state.roles == WEREWOLF,
            next_state.game_winner == 1,
            next_state.game_winner == 0,
        ).astype(jnp.float32)
        info = {
            "returns": rewards,
            "timestep": next_state.timestep,
            "game_winner": player_won,  # 1 for each player whose team has won
        }
        return next_state, obs, rewards, next_state.absorbing, next_state.done, info

    def observation_space(self) -> Discrete:
        return Discrete(self.obs_dim)

    def action_space(self) -> Discrete:
        return Discrete(self.num_actions)
