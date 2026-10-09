"""Leduc Hold'em env vs the exact game model in leduc_exploitability.

The model is OpenSpiel's leduc_poker, where player 0 always acts first. The
env draws the first player at random, so model player p (its position) is env
player (start + p) % 2.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bluffjax import make
from bluffjax.utils.game_utils import leduc_exploitability as leduc


@pytest.fixture(scope="module")
def env():
    return make("leduc_holdem")


@pytest.fixture(scope="module")
def step(env):
    return jax.jit(env.step_env)


def _env_state_for_deal(env, cards, start):
    """Env state dealing cards[p] to the player in position p."""
    state, _ = env.reset(jax.random.PRNGKey(0))
    agent_cards = [0, 0]
    for position, card in enumerate(cards):
        agent_cards[(start + position) % 2] = card
    rest = [c for c in range(6) if c not in cards]
    return state.replace(
        agent_cards=jnp.array(agent_cards, dtype=jnp.int32),
        shuffled_deck=jnp.array([*agent_cards, *rest], dtype=jnp.int32),
        start_player_idx=jnp.int32(start),
        current_player_idx=jnp.int32(start),
    )


def _with_public_card(env_state, card):
    """The env deals shuffled_deck[2] on the flop; put the model's chance outcome there."""
    deck = [int(c) for c in env_state.shuffled_deck]
    deck.remove(card)
    deck.insert(2, card)
    return env_state.replace(shuffled_deck=jnp.array(deck, dtype=jnp.int32))


def _walk(env, step, model_state, env_state, start, counts):
    """Follow every action and flop in both, checking they agree at each node."""
    if model_state.is_terminal():
        counts["terminal"] += 1
        return
    counts["decision"] += 1
    position = model_state.current_player
    assert int(env_state.current_player_idx) == (start + position) % 2
    legal = np.zeros(3, dtype=bool)
    legal[list(model_state.legal_actions())] = True
    np.testing.assert_array_equal(np.asarray(env.get_avail_actions(env_state)), legal)
    np.testing.assert_array_equal(
        np.asarray(env.obs_from_state(env_state)),
        leduc._infoset_key_to_obs(model_state.info_state_key(position)),
    )
    for action in model_state.legal_actions():
        child = model_state.child(action)
        flops = child.legal_actions() if child.is_chance_node() else (None,)
        for card in flops:
            model_next = child if card is None else child.child(card)
            before = env_state if card is None else _with_public_card(env_state, card)
            next_env, _, reward, absorbing, done, _ = step(jax.random.PRNGKey(1), before, action)
            if model_next.is_terminal():
                assert bool(done) and bool(absorbing.all())
                returns = model_next.returns()
                for p in (0, 1):
                    assert float(reward[(start + p) % 2]) == pytest.approx(returns[p])
            else:
                assert not bool(done)
                np.testing.assert_array_equal(np.asarray(reward), np.zeros(2))
            _walk(env, step, model_next, next_env, start, counts)


def _model_counts(state, counts):
    if state.is_terminal():
        counts["terminal"] += 1
        return
    if not state.is_chance_node():
        counts["decision"] += 1
    for action in state.legal_actions():
        _model_counts(state.child(action), counts)


def test_env_matches_exact_model_on_full_tree(env, step):
    """Acting player, legal actions, observation and returns at every node, both start players."""
    counts = {"decision": 0, "terminal": 0}
    for start in (0, 1):
        for cards in itertools.permutations(range(6), 2):
            model_state = leduc.initial_state().child(cards[0]).child(cards[1])
            _walk(env, step, model_state, _env_state_for_deal(env, cards, start), start, counts)
    # Every node of the model's tree was visited once per start player.
    expected = {"decision": 0, "terminal": 0}
    _model_counts(leduc.initial_state(), expected)
    assert counts == {k: 2 * v for k, v in expected.items()}


def test_start_player_is_random(env):
    states, _ = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0), 4000))
    np.testing.assert_array_equal(
        np.asarray(states.current_player_idx), np.asarray(states.start_player_idx)
    )
    assert 0.46 < float(states.start_player_idx.mean()) < 0.54


def test_observation_is_perfect_recall():
    """Two infosets share an observation only if they differ just in card suits."""
    by_obs = {}
    for key, legal in leduc._INFOSET_LIST:
        player, private_card, public_card, round1, round2 = key[:5]
        suit_free = (player, private_card // 2, -1 if public_card < 0 else public_card // 2, round1, round2)
        by_obs.setdefault(leduc._infoset_key_to_obs(key).tobytes(), set()).add((suit_free, legal))
    assert all(len(v) == 1 for v in by_obs.values())
    assert len(by_obs) == 288


def test_uniform_exploitability():
    assert leduc.exploitability_uniform() == pytest.approx(2.373611111111111)
