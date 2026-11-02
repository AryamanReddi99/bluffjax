"""
Exact evaluation of Leduc ReBeL (leduc_rebel.py) with other test-time settings.

For every seed's value network in {run_dir}/{stage}/value_net_{i}.msgpack,
computes the exact exploitability of ReBeL's test-time policy
(leduc_rebel.test_time_policy) with each number of CFR iterations per
subgame in --cfr-iters, at --precision. With --oracle K it instead uses the
exact value function (round-2 subgames solved with K CFR iterations), the
policy a perfect value network would give, which is the floor of the
test-time procedure at that number of iterations. Writes a JSON file.

    python bluffjax/examples/leduc/leduc_rebel_eval.py <run_dir> <out.json> --cfr-iters 1024 2048
    python bluffjax/examples/leduc/leduc_rebel_eval.py none <out.json> --oracle 1024 --cfr-iters 512 1024 2048
"""

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from bluffjax.examples.leduc import leduc_rebel as lr
from bluffjax.examples.run_utils import checkpoint_path, load_params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="run directory, or 'none' with --oracle")
    parser.add_argument("out")
    parser.add_argument("--stage", default="full")
    parser.add_argument("--cfr-iters", type=int, nargs="+", required=True)
    parser.add_argument("--precision", default="highest")
    parser.add_argument("--oracle", type=int, default=0, help="CFR iterations of the exact value function")
    parser.add_argument("--value-target", default="iterates")
    args = parser.parse_args()

    if args.oracle:
        value_fns = [lr.exact_value_fn(args.oracle, args.value_target)]
    else:
        config = OmegaConf.to_container(OmegaConf.load(f"{args.run_dir}/config.yaml"))
        network = lr.ValueNetwork(config["value_hidden_dim"], config["value_num_layers"])
        template = network.init(jax.random.PRNGKey(0), jnp.zeros(lr.PBS_DIM))
        value_fns = [
            lr.network_value_fn(
                network,
                load_params(checkpoint_path(args.run_dir, args.stage, "value_net", i), template),
            )
            for i in range(config["num_seeds"])
        ]
    results = []
    for iters in args.cfr_iters:
        t0 = time.time()
        expl = []
        for value_fn in value_fns:
            with jax.default_matmul_precision(args.precision):
                policy = jax.device_get(jax.jit(lambda: lr.test_time_policy(value_fn, iters))())
            expl.append(lr.evaluate_policy(policy)["exploitability"])
        entry = {
            "cfr_iters": iters,
            "precision": args.precision,
            "oracle": args.oracle,
            "stage": args.stage,
            "exploitability": expl,
            "mean": float(np.mean(expl)),
            "sd": float(np.std(expl, ddof=1)) if len(expl) > 1 else 0.0,
            "seconds": time.time() - t0,
        }
        results.append(entry)
        print(json.dumps(entry), flush=True)
        with open(args.out, "w") as f:
            json.dump({"run_dir": args.run_dir, "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
