# BluffJAX in the browser

Play all ten BluffJAX games against bots in a web page, running the real
BluffJAX JAX code client-side. Once GitHub Pages serves this folder, the demo
lives at https://aryamanreddi99.github.io/bluffjax/.

Each game's `reset`, `step_env` and `get_avail_actions` are exported with
`jax.export` to StableHLO and compiled to WebAssembly in the page by
[whlo](https://github.com/noahfarr/whlo). There is no server and no Python at
play time.

## Run it locally

```sh
cd docs && uv run python -m http.server 8000
# open http://localhost:8000/        (deep link to a game: /#werewolf)
```

You always sit in seat 0. The other seats are simple heuristic bots in
`index.html`, not trained policies.

## Files

- `index.html`: the page, the renderer (each game is drawn from its named
  state fields) and the bots.
- `games/`: the exported StableHLO modules plus `index.json`, which names
  every state leaf.
- `vendor/whlo/`: a build of whlo (Apache-2.0, see `vendor/whlo/LICENSE`).
- `export_games.py`: regenerates `games/` from the environments.
- `verify/`: checks whlo against JAX.

## After changing an environment

Re-export so the demo picks up the change. `uv run` uses the repo's locked
environment, so there is nothing else to install:

```sh
cd docs
JAX_PLATFORMS=cpu uv run python export_games.py          # all games, or: ... kuhn_poker bluff
```

Then check the exports still match JAX:

```sh
uv run python verify/make_ref.py kuhn_poker 400   # JAX rollout with random legal actions
node verify/check.mjs kuhn_poker                  # replay through whlo and compare
```

`check.mjs` compares the full state, rewards, done flags and legal-action
masks at every step. All ten games currently match JAX bit for bit over 400
random legal moves each.

A new environment also needs a renderer and a bot in `index.html`; the
existing games show the pattern.

Standalone copy: https://github.com/noahfarr/bluffjax.wHLO
