# 🌙 `Zodiax` ✨

[![PyPI version](https://badge.fury.io/py/zodiax.svg)](https://badge.fury.io/py/zodiax)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![integration](https://github.com/LouisDesdoigts/zodiax/actions/workflows/tests.yml/badge.svg)](https://github.com/LouisDesdoigts/zodiax/actions/workflows/tests.yml)
[![Documentation](https://github.com/LouisDesdoigts/zodiax/actions/workflows/documentation.yml/badge.svg)](https://louisdesdoigts.github.io/zodiax/)

Contributors: [Louis Desdoigts](https://github.com/LouisDesdoigts)


`Zodiax` is a lightweight extension to the object-oriented `jax` package `equinox`, designed to to simplify working with pytrees for scientific data analysis, adding extra functionality to make working with optimisation and inference libraries like `optax` and `NumPyro` easy. If you are unfamiliar with `Jax` and `equinox` we strongly reccomend reading the [Jax 101 tutorials](https://jax.readthedocs.io/en/latest/jax-101/index.html) and the [equinox docs](https://docs.kidger.site/equinox/) to get an idea. 

`Zodiax` provides classes that extend the `equinox.Module` class with a series of convenience methods that make working with and optimising complex nested class structures easy. It was built to facilitate [`dLux`](https://louisdesdoigts.github.io/dLux/), a fully differentiable optical modelling framework, and spun out into its own package to aid others in building the next generation of powerful scientific programming tools that harness automatic differentiation.

To get started, go to the Tutorials section and have a look!

---

## Installation

`Zodiax` is hosted on PyPI: 

```
pip install zodiax
```

---

## Citation

If you use `zodiax` in your work, please cite the `dLux` paper: currently Desdoigts et al, in prep.

