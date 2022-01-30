from typing import Iterable, Generator
import jax
from chex import ArrayTree


def _device_put_sharded(sharded_tree, devices):
    leaves, treedef = jax.tree_flatten(sharded_tree)
    n = leaves[0].shape[0]
    return jax.device_put_sharded(
        [jax.tree_unflatten(treedef, [l[i] for l in leaves]) for i in range(n)], devices
    )


def double_buffer(ds: Iterable[ArrayTree]) -> Generator[ArrayTree, None, None]:
    """Keeps at least two batches on the accelerator.

    The current GPU allocator design reuses previous allocations. For a training
    loop this means batches will (typically) occupy the same region of memory as
    the previous batch. An issue with this is that it means we cannot overlap a
    host->device copy for the next batch until the previous step has finished and
    the previous batch has been freed.

    By double buffering we ensure that there are always two batches on the device.
    This means that a given batch waits on the N-2'th step to finish and free,
    meaning that it can allocate and copy the next batch to the accelerator in
    parallel with the N-1'th step being executed.

    Args:
      ds: Iterable of batches of numpy arrays.

    Yields:
      Batches of sharded device arrays.
    """
    batch = None
    devices = jax.local_devices()
    for next_batch in ds:
        assert next_batch is not None
        if len(devices) > 1:
            next_batch = _device_put_sharded(next_batch, devices)
        else:
            next_batch = jax.device_put(next_batch, devices[0])
        if batch is not None:
            yield batch
        batch = next_batch
    if batch is not None:
        yield batch
