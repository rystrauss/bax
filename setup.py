from setuptools import setup

with open("README.md", "r") as fp:
    long_description = fp.read()

setup(
    name="bax",
    version="0.2.0",
    packages=["bax"],
    url="https://github.com/rystrauss/bax",
    license="LICENSE",
    author="Ryan Strauss",
    author_email="ryanrstrauss@icloud.com",
    description="A flexible trainer interface for Jax and Haiku.",
    python_requires=">=3.7",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "dm-haiku>=0.0.6",
        "jax>=0.3.13",
        "optax>=0.1.2",
        "jmp>=0.0.2",
        "chex>=0.1.3",
        "keras>=2.9.0",
        "tensorflow>=2.9.0",
        "rich>=12.4.1",
    ],
)
