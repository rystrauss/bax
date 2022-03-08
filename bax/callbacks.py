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
    def __init__(self, path: str, track: str = "val_loss", objective: str = "min"):
        self._path = path
        self._track = track
        self._objective = objective
        self._best = float("inf") if objective == "min" else -float("inf")

    def on_validation_end(
        self, train_state: TrainState, step: int, logs: Dict[str, Any]
    ):
        if (self._objective == "min" and logs[self._track] < self._best) or (
            self._objective == "max" and logs[self._track] > self._best
        ):
            self._best = logs[self._track]

            if isinstance(jax.tree_leaves(train_state.params)[0], ShardedDeviceArray):
                train_state = jax.tree_map(lambda x: x[0], train_state)

            train_state = jax.device_get(train_state)

            with open(self._path, "wb") as fp:
                pickle.dump(train_state, fp)


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
