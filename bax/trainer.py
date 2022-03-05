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
import jmp
import optax
import tensorflow as tf
from chex import PRNGKey, ArrayTree, Scalar, Array
from keras.metrics import Mean
from tqdm import tqdm

from bax.data import double_buffer

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
    loss_scale: jmp.LossScale
    ema_params: Optional[hk.Params] = None


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
        mp_policy: A `jmp.Policy` that defines the mixed precision policy to use in
            for training steps.
        skip_nonfinite_updates: If True, then updates with non-finite gradients will
            be skipped.
        loss_scale: A `jmp.LossScale` object, which defines potential loss scaling to
            be applied when calculating gradients.
        gradient_skipping_threshold: If specified, then updates with a global gradient
            norm that is greater than this will be skipped.
        ema_rate: If provided the exponential moving average, with this rate, of the
            parameters will be tracked and included in the train state.
        use_ema_for_eval: If True and `ema_rate` is not None, then the EMA parameters
            will be used for the validation data.
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
        mp_policy: Optional[jmp.Policy] = None,
        skip_nonfinite_updates: bool = False,
        loss_scale: Optional[jmp.LossScale] = None,
        gradient_skipping_threshold: Optional[float] = None,
        ema_rate: Optional[float] = None,
        use_ema_for_eval: bool = False,
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
        self._mp_policy = mp_policy or jmp.get_policy("full")
        self._skip_nonfinite_updates = skip_nonfinite_updates
        self._loss_scale = loss_scale or jmp.NoOpLossScale()
        self._gradient_skipping_threshold = gradient_skipping_threshold
        self._ema_rate = ema_rate
        self._use_ema_for_eval = use_ema_for_eval

        if isinstance(loss_scale, jmp.DynamicLossScale) and not skip_nonfinite_updates:
            print(
                "Warning: using jmp.DynamicLossScale but skip_nonfinite_updates is "
                "False. Consider setting skip_nonfinite_updates=True."
            )

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

            self._grad_fn = jax.pmap(
                self._grad_step,
                axis_name=self.cross_replica_axis,
                in_axes=(0, 0, None, 0),
                donate_argnums=0,
            )
            self._eval_fn = jax.pmap(
                self._eval_step,
                axis_name=self.cross_replica_axis,
                in_axes=(0, 0, None, 0),
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
        grads = self._mp_policy.cast_to_compute(grads)
        grads = train_state.loss_scale.unscale(grads)

        if self._num_devices > 1:
            grads = jax.lax.pmean(grads, axis_name=self.cross_replica_axis)

        grads = self._mp_policy.cast_to_param(grads)

        updates, new_opt_state = self._optimizer.update(
            grads, train_state.opt_state, trainable_params
        )
        new_trainable_params = optax.apply_updates(trainable_params, updates)

        loss_scale = train_state.loss_scale
        if self._skip_nonfinite_updates:
            grads_finite = jmp.all_finite(grads)
            loss_scale = train_state.loss_scale.adjust(grads_finite)
            new_trainable_params, new_state, new_opt_state = jmp.select_tree(
                grads_finite,
                (new_trainable_params, new_state, new_opt_state),
                (trainable_params, train_state.state, train_state.opt_state),
            )
            aux["mp_grads_finite"] = grads_finite

        if not isinstance(self._loss_scale, jmp.NoOpLossScale):
            aux["mp_loss_scale"] = loss_scale.loss_scale

        should_skip = False
        if self._gradient_skipping_threshold is not None:
            should_skip = optax.global_norm(grads) > self._gradient_skipping_threshold
            new_trainable_params, new_state, new_opt_state = jmp.select_tree(
                should_skip,
                (trainable_params, train_state.state, train_state.opt_state),
                (new_trainable_params, new_state, new_opt_state),
            )

        new_params = hk.data_structures.merge(
            new_trainable_params, non_trainable_params
        )

        new_state, aux = jmp.cast_to_full((new_state, aux))

        if self._num_devices > 1:
            aux = jax.lax.pmean(aux, axis_name=self.cross_replica_axis)
            loss = jax.lax.pmean(loss, axis_name=self.cross_replica_axis)

        if self._ema_rate is not None and not should_skip:
            new_ema_params = jax.tree_multimap(
                lambda e, p: e * self._ema_rate + (1 - self._ema_rate) * p,
                train_state.ema_params,
                new_params,
            )
        else:
            new_ema_params = train_state.ema_params

        new_train_state = TrainState(
            new_params, new_state, new_opt_state, loss_scale, new_ema_params
        )
        return new_train_state, loss, aux

    def _eval_step(
        self, train_state: TrainState, key: PRNGKey, step: int, batch: ArrayTree
    ) -> Tuple[Scalar, Mapping[str, Scalar]]:
        fn = self._validation_fn or self._loss
        if self._ema_rate is not None and self._use_ema_for_eval:
            params = train_state.ema_params
        else:
            params = train_state.params
        (loss, aux), _ = fn.apply(params, train_state.state, key, step, False, batch)
        if self._num_devices > 1:
            aux = jax.lax.pmean(aux, axis_name=self.cross_replica_axis)
            loss = jax.lax.pmean(loss, axis_name=self.cross_replica_axis)
        return loss, aux

    def _get_pmap_keys(self) -> Array:
        return jnp.squeeze(jax.random.split(self._prng.next(), self._num_devices))

    def _get_initial_train_state(
        self, key, init_batch, initial_params, initial_state, initial_opt_state
    ):
        init_params, init_state = self._loss.init(key, 0, True, init_batch)

        if initial_params is not None:
            init_params = hk.data_structures.merge(init_params, initial_params)

        if initial_state is not None:
            init_state = hk.data_structures.merge(init_state, initial_state)

        trainable_params, _ = hk.data_structures.partition(
            self._trainable_predicate, init_params
        )
        init_opt_state = initial_opt_state or self._optimizer.init(trainable_params)

        init_params = self._mp_policy.cast_to_param(init_params)
        ema_params = None if self._ema_rate is None else init_params

        return TrainState(
            params=init_params,
            state=init_state,
            opt_state=init_opt_state,
            loss_scale=self._loss_scale,
            ema_params=ema_params,
        )

    def fit(
        self,
        train_dataset: tf.data.Dataset,
        steps: int,
        val_dataset: Optional[tf.data.Dataset] = None,
        validation_freq: int = 1000,
        callbacks: Optional[List["Callback"]] = None,
        initial_params: Optional[hk.Params] = None,
        initial_state: Optional[hk.State] = None,
        initial_opt_state: Optional[optax.OptState] = None,
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
            initial_opt_state: Optional initial optimizer state. Can be used to resume
                training from a previous run, for example.
            verbose: The verbosity level.

        Returns:
            The final TrainState.
        """
        init_batch = next(train_dataset.take(1).as_numpy_iterator())

        if self._num_devices <= 1:
            train_state = self._get_initial_train_state(
                self._prng.next(),
                init_batch,
                initial_params,
                initial_state,
                initial_opt_state,
            )

            byte_size = hk.data_structures.tree_bytes(train_state.params)
            num_params = hk.data_structures.tree_size(train_state.params)
        else:
            keys = jnp.asarray([self._prng.next()] * self._num_devices)
            train_state = jax.pmap(
                self._get_initial_train_state,
                axis_name=self.cross_replica_axis,
                in_axes=(0, None, None, None, None),
            )(keys, init_batch, initial_params, initial_state, initial_opt_state)

            params = jax.tree_map(lambda x: x[0], train_state.params)
            byte_size = hk.data_structures.tree_bytes(params)
            num_params = hk.data_structures.tree_size(params)

        print(f"Total Parameters: {num_params}, {byte_size / 1e6:.2f}MB")

        metrics = defaultdict(Mean)

        callbacks = callbacks or []

        if self._num_devices > 1:
            train_dataset = train_dataset.batch(self._num_devices, drop_remainder=True)
            if val_dataset is not None:
                val_dataset = val_dataset.batch(self._num_devices, drop_remainder=True)

        train_iter = train_dataset.repeat().as_numpy_iterator()

        if jax.default_backend() == "gpu":
            train_iter = double_buffer(train_iter)

        pbar = tqdm(desc="Training", disable=verbose == 0, total=steps)

        for step, batch in enumerate(train_iter):
            train_state, loss, aux = self._grad_fn(
                train_state, self._get_pmap_keys(), step, batch
            )

            if self._num_devices > 1:
                aux = jax.tree_map(lambda x: x[0], aux)
                loss = jax.tree_map(lambda x: x[0], loss)

            aux = jax.device_get(aux)
            loss = jax.device_get(loss)

            metrics["loss"].update_state(loss)
            for k, v in aux.items():
                metrics[k].update_state(v)

            if val_dataset is not None and step % validation_freq == 0:
                val_iter = val_dataset.as_numpy_iterator()
                if jax.default_backend() == "gpu":
                    val_iter = double_buffer(val_iter)

                for val_batch in val_iter:
                    val_loss, val_aux = self._eval_fn(
                        train_state,
                        self._get_pmap_keys(),
                        step,
                        val_batch,
                    )

                    for callback in callbacks:
                        callback.on_validation_step(
                            train_state, self._get_pmap_keys(), val_batch
                        )

                    if self._num_devices > 1:
                        val_aux = jax.tree_map(lambda x: x[0], val_aux)
                        val_loss = jax.tree_map(lambda x: x[0], val_loss)

                    val_aux = jax.device_get(val_aux)
                    val_loss = jax.device_get(val_loss)

                    metrics["val_loss"].update_state(val_loss)
                    for k, v in val_aux.items():
                        metrics[f"val_{k}"].update_state(v)

                logs = {k: v.result().numpy().item() for k, v in metrics.items()}

                print_string = f"[Step {step}]"

                for k, v in logs.items():
                    print_string += f" -- {k}: {v:.3f}"

                pbar.write(print_string)

                for callback in callbacks:
                    callback.on_validation_end(train_state, step, logs)

                for v in metrics.values():
                    v.reset_state()

            if step >= steps:
                break

            pbar.update()

        if self._num_devices > 1:
            train_state = jax.tree_map(lambda x: x[0], train_state)

        return jax.device_get(train_state)
