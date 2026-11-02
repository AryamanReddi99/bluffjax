"""
Re-evaluate saved Kuhn ReBeL value networks (kuhn_rebel.py checkpoints).

For every seed's value network in {run_dir}/{stage}/value_net_{i}.msgpack,
computes the exact exploitability of ReBeL's test-time policy
(kuhn_rebel.test_time_policy) with each number of CFR iterations per subgame
in --cfr-iters, at the given matmul precision (the training script evaluates
inside its jit at JAX's default precision, which is TF32 on Ampere GPUs; the
test-time policy is sensitive to it). Writes a JSON file with one entry per
(cfr_iters, precision).

    python bluffjax/examples/kuhn/kuhn_rebel_eval.py <run_dir> <out.json> \
        --cfr-iters 512 1024 --precision default highest
"""

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from bluffjax.examples.kuhn import kuhn_rebel as kr
from bluffjax.examples.run_utils import checkpoint_path, load_params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("out")
    parser.add_argument("--stage", default="full")
    parser.add_argument("--cfr-iters", type=int, nargs="+", default=None)
    parser.add_argument("--precision", nargs="+", default=["highest"])
    args = parser.parse_args()

    config = OmegaConf.to_container(OmegaConf.load(f"{args.run_dir}/config.yaml"))
    network = kr.ValueNetwork(hidden_dim=config["value_hidden_dim"])
    template = network.init(jax.random.PRNGKey(0), jnp.zeros(kr.PBS_DIM))
    structure = kr.subgame_structure(config["max_depth"])
    params = [
        load_params(checkpoint_path(args.run_dir, args.stage, "value_net", i), template)
        for i in range(config["num_seeds"])
    ]
    results = []
    for iters in args.cfr_iters or [config["cfr_iters"]]:
        for precision in args.precision:
            t0 = time.time()
            expl = []
            for p in params:
                with jax.default_matmul_precision(precision):
                    policy = np.asarray(
                        kr.test_time_policy(kr.network_value_fn(network, p), structure, iters)
                    )
                expl.append(kr.policy_exploitability(policy))
            entry = {
                "cfr_iters": iters,
                "precision": precision,
                "stage": args.stage,
                "exploitability": expl,
                "mean": float(np.mean(expl)),
                "sd": float(np.std(expl, ddof=1)),
                "seconds": time.time() - t0,
            }
            results.append(entry)
            print(json.dumps(entry), flush=True)
            with open(args.out, "w") as f:
                json.dump({"run_dir": args.run_dir, "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
