from setuptools import setup

setup(
    name="bax",
    version="0.1",
    packages=["bax"],
    url="https://github.com/rystrauss/bax",
    license="",
    author="Ryan Strauss",
    author_email="ryanrstrauss@icloud.com",
    description="A flexible trainer interface for Jax and Haiku.",
    python_requires=">=3.8",
    install_requires=[
        "dm-haiku>=0.0.5",
        "jax>=0.2.24",
        "optax>=0.0.9",
        "chex>=0.0.8",
        "keras>=2.6.0",
        "tensorflow>=2.6"
    ],
)
