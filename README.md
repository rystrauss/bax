# Bax

Bax, short for "boilerplate jax", is a small library that provides a flexible trainer
interface for Jax.

Bax is rather strongly opinionated in a few ways. First, it is designed for use with the
[Haiku](https://github.com/deepmind/dm-haiku) neural network library and is not
compatible with e.g. [Flax](https://github.com/google/flax). Second, Bax assumes that
data will be provided as a `tf.data.Dataset`. The goal of this library is not to be
widely compatible and high-level (like [Elegy](https://github.com/poets-ai/elegy)).

If you are okay with making the above assumptions, then Bax will hopefully make your
life much easier by implementing the boilerplate code involved in neural network
training loops.

Please note that this library has not yet been extensively tested.

## Installation

You can install Bax via pip:

```
pip install bax
```

## Usage

Below are some simple examples that illustrate how to use Bax.

### MNIST Classification

```python
import optax
import tensorflow_datasets as tfds
import haiku as hk
import jax.numpy as jnp
import jax

from bax.trainer import Trainer


# Use TensorFlow Datasets to get our MNIST data.
train_ds = tfds.load("mnist", split="train").batch(32, drop_remainder=True)
test_ds = tfds.load("mnist", split="test").batch(32, drop_remainder=True)

# The loss function that we want to minimize.
def loss_fn(step, is_training, batch):
    model = hk.Sequential([hk.Flatten(), hk.nets.MLP([128, 128, 10])])

    preds = model(batch["image"] / 255.0)
    labels = jax.nn.one_hot(batch["label"], 10)

    loss = jnp.mean(optax.softmax_cross_entropy(preds, labels))
    accuracy = jnp.mean(jnp.argmax(preds, axis=-1) == batch["label"])

    # The first returned value is the loss, which is what will be minimized by the
    # trainer. The second value is a dictionary that can contain other metrics you
    # might be interested in (or, it can just be empty).
    return loss, {"accuracy": accuracy}

trainer = Trainer(loss=loss_fn, optimizer=optax.adam(0.001))

# Run the training loop. Metrics will be printed out each time the validation
# dataset is evaluated (in this case, every 1000 steps).
trainer.fit(train_ds, steps=10000, val_dataset=test_ds, validation_freq=1000)
```
