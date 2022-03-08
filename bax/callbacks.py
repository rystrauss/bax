import pickle
from typing import Dict, Any, Optional, Sequence

import jax
from chex import PRNGKey, ArrayTree
from jax.interpreters.pxla import ShardedDeviceArray
from optax import Schedule

from bax.trainer import TrainState


class Callback:
    def on_train_step(self, step: int, logs: Dict[str, Any]):
        pass

    def on_validation_step(
        self, train_state: TrainState, key: PRNGKey, batch: ArrayTree
    ):
        pass

    def on_validation_end(
        self, train_state: TrainState, step: int, logs: Dict[str, Any]
    ):
        pass


class LearningRateLoggerCallback(Callback):
    def __init__(self, schedule: Schedule):
        self._schedule = schedule

    def on_validation_end(
        self, train_state: TrainState, step: int, logs: Dict[str, Any]
    ):
        logs["learning_rate"] = self._schedule(step)


class CheckpointCallback(Callback):
    def __init__(
        self,
        path: str,
        track: str = "val_loss",
        objective: str = "min",
        save_freq: int = 1,
    ):
        self._path = path
        self._track = track
        self._objective = objective
        self._save_freq = save_freq
        self._best = float("inf") if objective == "min" else -float("inf")

        self._count = 0
        self.best_checkpoint = None

    def on_validation_end(
        self, train_state: TrainState, step: int, logs: Dict[str, Any]
    ):
        if isinstance(jax.tree_leaves(train_state.params)[0], ShardedDeviceArray):
            train_state = jax.tree_map(lambda x: x[0], train_state)

        train_state = jax.device_get(train_state)
        checkpoint_path = self._path.format(**logs)

        if (self._objective == "min" and logs[self._track] < self._best) or (
            self._objective == "max" and logs[self._track] > self._best
        ):
            self._best = logs[self._track]
            self.best_checkpoint = checkpoint_path

            with open(checkpoint_path, "wb") as fp:
                pickle.dump(train_state, fp)

        if self._count % self._save_freq == 0:
            with open(checkpoint_path, "wb") as fp:
                pickle.dump(train_state, fp)

        self._count += 1


class WandbCallback(Callback):
    def __init__(
        self,
        run: "wandb.sdk.wandb_run.Run",
        train_step_metrics: Optional[Sequence[str]] = None,
    ):
        self._run = run
        self._train_step_metrics = train_step_metrics

    def on_train_step(self, step: int, logs: Dict[str, Any]):
        if self._train_step_metrics is not None:
            to_log = {}

            for k in self._train_step_metrics:
                to_log[k] = logs[k]

            self._run.log(logs, step=step)

    def on_validation_end(
        self, train_state: TrainState, step: int, logs: Dict[str, Any]
    ):
        self._run.log(logs, step=step)
