"""
Class to spawn separate wandb processes so that multiple runs
started from one python process can be logged separately
(e.g. multiple seeds from one script)

Wandb typically only allows one process to spawn for each python
process, so this allows us to spawn multiple processes for each
wandb run, place them in a queue awaiting inputs, and log to them.

The workers are started with the "spawn" method: forking a process after JAX
has started its threads can deadlock. They are daemons, so a crashed training
run can't hang on them; finish() joins them so every queued log is sent.
With spawn, a script that creates the logger must guard its entry point
with `if __name__ == "__main__":` (all training scripts do).
"""

import multiprocessing as mp

import wandb

from bluffjax.utils.paths import REPO_ROOT

_CTX = mp.get_context("spawn")


def worker(
    project: str, group: str, job_type: str, name: str, config: dict, mode: str, queue
) -> None:
    wandb.init(
        project=project,
        group=group,
        job_type=job_type,
        name=name,
        config=config,
        mode=mode,
        dir=str(REPO_ROOT),
    )
    try:
        while True:
            data = queue.get()
            if data is None:
                # Sentinel to end logging
                break
            # Log the received data to W&B
            wandb.log(data)
    finally:
        # Ensure W&B run is properly closed
        wandb.finish()


class WandbMultiLogger:
    """
    Keeps a pair of dictionaries indexed by seed indices (0,1,etc.)
    self.processes contains references for each wandb process and
    self.queues keeps a queue for each process indexed by the same key (seed no.)
    """

    def __init__(
        self,
        project: str,
        group: str,
        job_type: str,
        config: dict,
        mode: str,
        seed: int,
        num_seeds: int,
    ):
        wandb_settings = {
            "project": project,
            "group": group,
            "job_type": job_type,
            "config": config,
            "mode": mode,
        }
        self.processes = {}
        self.queues = {}
        for i in range(num_seeds):
            q = _CTX.Queue()
            self.queues[i] = q
            wandb_settings.update({"name": f"{seed}_{i}", "queue": q})
            p = _CTX.Process(target=worker, kwargs=wandb_settings, daemon=True)
            p.start()
            self.processes[i] = p

    def log(self, seed: int, data_dict: dict):
        self.queues[seed].put(data_dict)

    def finish(self):
        for seed in self.processes.keys():
            self.queues[seed].put(None)
        for seed in self.processes.keys():
            self.processes[seed].join()
