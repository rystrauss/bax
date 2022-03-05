import os


def set_jax_memory_preallocation(value: bool):
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" if value else "false"


def set_tf_memory_preallocation(value: bool):
    from tensorflow import config

    gpus = config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                config.experimental.set_memory_growth(gpu, value)
        except RuntimeError as e:
            print(e)
