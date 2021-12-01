import random
from collections import defaultdict
from typing import (
    NamedTuple,
    Callable,
    Tuple,
    Optional,
    Mapping,
    List,
    Dict,
)

import haiku as hk
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
from chex import PRNGKey, ArrayTree, Scalar, Array
from keras.metrics import Mean
from keras.utils.generic_utils import Progbar

# A function that can be optimized by the Trainer class. The function accepts
# the training step number, a boolean indicating whether or not training mode is
# enabled (as opposed to evaluation), and batch of data as inputs, and outputs a scalar
# loss as well as a dictionary of additional metrics to log.
LossFunction = Callable[[int, bool, ArrayTree], Tuple[Scalar, Mapping[str, Scalar]]]


class TrainState(NamedTuple):
    """Container for model parameters and state, and optimizer state."""
    params: hk.Params
    state: hk.State
    opt_state: optax.OptState


class Trainer:
    """An object that optimizes Jax/Haiku loss functions via gradient descent.

    This class implements a standard gradient descent training loop as well as
    functionality for metric logging, parallelization, and non-trainable parameters.

    Args:
        loss: The loss function to minimize.
        optimizer: The optimizer with which to minimize the loss.
        validation_fn: An optional function that can be different from the loss
            function and will be used during validation (if validation data is
            provided). By default, the loss function is used during validation.
        trainable_predicate: A function that indicates which parameters are trainable.
            By default, all parameters are trainable. See documentation for
            `hk.data_structures.partition` for more details.
        run_eagerly: If true, computations will not be JIT compiled.
        num_devices: The number of devices over which to parallelize computations.
            For example, if `num_devices=2`, then two minibatches will be processed at
            the same time, each on its own GPU/TPU, and gradients/metrics will be
            reduced across devices (i.e. the batch size is effectively doubled).
            By default, only a single devices is used.
        shard_validation: Whether or not to parallelize validation, assuming
            `num_devices` is set to a value greater than 2. By default this is False.
        seed: An optional random seed used to initialize the Trainer's random number
            generation.
    """
    cross_replica_axis: str = "r"

    def __init__(
        self,
        loss: LossFunction,
        optimizer: optax.GradientTransformation,
        validation_fn: Optional[LossFunction] = None,
        trainable_predicate: Callable[[str, str, Array], bool] = None,
        run_eagerly: bool = False,
        num_devices: Optional[int] = None,
        shard_validation: bool = False,
        seed: Optional[int] = None,
    ):
        self._loss = hk.transform_with_state(loss)
        self._optimizer = optimizer
        self._validation_fn = (
            hk.transform_with_state(validation_fn)
            if validation_fn is not None
            else None
        )
        self._trainable_predicate = trainable_predicate or (lambda *args: True)
        self._shard_validation = shard_validation

        local_device_count = jax.local_device_count()
        num_devices = num_devices or local_device_count
        if num_devices > local_device_count:
            raise ValueError(
                f"num_devices is {num_devices}, but there are only "
                f"{local_device_count} available devices."
            )
        self._num_devices = num_devices

        if self._num_devices <= 1:
            self._grad_fn = self._grad_step if run_eagerly else jax.jit(self._grad_step)
            self._eval_fn = self._eval_step if run_eagerly else jax.jit(self._eval_step)
        else:
            if run_eagerly:
                print(
                    "Warning: run_eagerly=True but num_devices > 1, so computation "
                    "will still be JIT compiled. Set num_devices=1 to allow for "
                    "eager execution."
                )

            in_axes = (None, 0, None, 0)
            self._grad_fn = jax.pmap(
                self._grad_step,
                axis_name=self.cross_replica_axis,
                in_axes=in_axes,
                out_axes=(None, 0, 0),
                donate_argnums=0,
            )

            if shard_validation:
                self._eval_fn = jax.pmap(
                    self._eval_step,
                    axis_name=self.cross_replica_axis,
                    in_axes=in_axes,
                    out_axes=(0, 0),
                )
            else:
                self._eval_fn = (
                    self._eval_step if run_eagerly else jax.jit(self._eval_step)
                )

        self._prng = hk.PRNGSequence(seed or random.randint(0, int(2e9)))

    def _partitioned_loss(
        self,
        trainable_params: hk.Params,
        non_trainable_params: hk.Params,
        state: hk.State,
        key: PRNGKey,
        step: int,
        batch: ArrayTree,
    ) -> Tuple[Scalar, Tuple[Dict[str, Scalar], hk.State]]:
        params = hk.data_structures.merge(trainable_params, non_trainable_params)
        (loss, aux), new_state = self._loss.apply(params, state, key, step, True, batch)
        return loss, (aux, new_state)

    def _grad_step(
        self, train_state: TrainState, key: PRNGKey, step: int, batch: ArrayTree
    ) -> Tuple[TrainState, Scalar, Mapping[str, Scalar]]:
        trainable_params, non_trainable_params = hk.data_structures.partition(
            self._trainable_predicate, train_state.params
        )
        (loss, (aux, new_state)), grads = jax.value_and_grad(
            self._partitioned_loss, has_aux=True
        )(trainable_params, non_trainable_params, train_state.state, key, step, batch)
        if self._num_devices > 1:
            grads = jax.lax.pmean(grads, axis_name=self.cross_replica_axis)
        updates, new_opt_state = self._optimizer.update(
            grads, train_state.opt_state, trainable_params
        )
        new_trainable_params = optax.apply_updates(trainable_params, updates)
        new_params = hk.data_structures.merge(
            new_trainable_params, non_trainable_params
        )

        return TrainState(new_params, new_state, new_opt_state), loss, aux

    def _eval_step(
        self, train_state: TrainState, key: PRNGKey, step: int, batch: ArrayTree
    ) -> Tuple[Scalar, Mapping[str, Scalar]]:
        fn = self._validation_fn or self._loss
        (loss, aux), _ = fn.apply(
            train_state.params, train_state.state, key, step, False, batch
        )
        return loss, aux

    def _get_pmap_keys(self) -> Array:
        return jnp.squeeze(jax.random.split(self._prng.next(), self._num_devices))

    def _get_initial_params_and_state(
        self, dataset: tf.data.Dataset
    ) -> Tuple[hk.Params, hk.State]:
        init_batch = next(dataset.as_numpy_iterator())

        if self._num_devices <= 1:
            return self._loss.init(self._prng.next(), 0, True, init_batch)

        return jax.pmap(
            self._loss.init,
            axis_name=self.cross_replica_axis,
            in_axes=(0, None, None, None),
            out_axes=(None, None),
        )(self._get_pmap_keys(), 0, True, init_batch)

    def fit(
        self,
        train_dataset: tf.data.Dataset,
        steps: int,
        val_dataset: Optional[tf.data.Dataset] = None,
        validation_freq: int = 1000,
        callbacks: Optional[List["Callback"]] = None,
        initial_params: Optional[hk.Params] = None,
        initial_state: Optional[hk.State] = None,
        opt_state: Optional[optax.OptState] = None,
        verbose: int = 1,
    ) -> TrainState:
        """Runs the training loop using the provided dataset.

        Args:
            train_dataset: The dataset to use for training.
            steps: The number of optimization steps to perform.
            val_dataset: An optional dataset to be used for validation.
            validation_freq: The frequency with which the validation dataset will be
                evaluated.
            callbacks: An optional list of callbacks apply during validation.
            initial_params: Optional parameter values that can be provided to override
                the randomly initialized parameters.
            initial_state: Optional state values that can be provided to override
                the default initialized state.
            opt_state: Optional initial optimizer state. Can be used to resume
                training from a previous run, for example.
            verbose: The verbosity level.

        Returns:
            The final TrainState.
        """
        init_params, init_state = self._get_initial_params_and_state(train_dataset)

        if initial_params is not None:
            init_params = hk.data_structures.merge(init_params, initial_params)

        if initial_state is not None:
            init_state = hk.data_structures.merge(init_state, initial_state)

        trainable_params, _ = hk.data_structures.partition(
            self._trainable_predicate, init_params
        )
        init_opt_state = opt_state or self._optimizer.init(trainable_params)

        train_state = TrainState(
            params=init_params, state=init_state, opt_state=init_opt_state
        )

        metrics = defaultdict(Mean)
        pbar = Progbar(steps, verbose=verbose)

        callbacks = callbacks or []

        if self._num_devices > 1:
            train_dataset = train_dataset.batch(self._num_devices, drop_remainder=True)
            if self._shard_validation:
                val_dataset = val_dataset.batch(self._num_devices, drop_remainder=True)

        train_iter = train_dataset.repeat().as_numpy_iterator()

        for step, batch in enumerate(train_iter):
            train_state, loss, aux = self._grad_fn(
                train_state, self._get_pmap_keys(), step, batch
            )

            metrics["loss"].update_state(loss)
            for k, v in aux.items():
                metrics[k].update_state(v)

            pbar.update(
                step,
                [
                    (n, m.result())
                    for n, m in metrics.items()
                    if not n.startswith("val_")
                ],
            )

            if val_dataset is not None and step % validation_freq == 0:
                for val_batch in val_dataset.as_numpy_iterator():
                    val_loss, val_aux = self._eval_fn(
                        train_state,
                        self._get_pmap_keys()
                        if self._shard_validation
                        else self._prng.next(),
                        step,
                        val_batch,
                    )

                    metrics["val_loss"].update_state(val_loss)
                    for k, v in val_aux.items():
                        metrics[f"val_{k}"].update_state(v)

                    for callback in callbacks:
                        callback.on_validation_step(
                            train_state, self._prng.next(), val_batch
                        )

                logs = {k: v.result().numpy().item() for k, v in metrics.items()}

                for callback in callbacks:
                    callback.on_validation_end(train_state, step, logs)

                for v in metrics.values():
                    v.reset_state()

            if step >= steps:
                break

        return jax.device_get(train_state)
