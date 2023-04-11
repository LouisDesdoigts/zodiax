<!-- <h1 align='center'>Zodiax</h1> -->
# Zodiax

[![PyPI version](https://badge.fury.io/py/zodiax.svg)](https://badge.fury.io/py/zodiax)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![integration](https://github.com/LouisDesdoigts/zodiax/actions/workflows/tests.yml/badge.svg)](https://github.com/LouisDesdoigts/zodiax/actions/workflows/tests.yml)
[![Documentation](https://github.com/LouisDesdoigts/zodiax/actions/workflows/documentation.yml/badge.svg)](https://louisdesdoigts.github.io/zodiax/)

---

[_Zodiax_](https://github.com/LouisDesdoigts/zodiax) is a lightweight extension to the object-oriented [_Jax_](https://github.com/google/jax) framework [_Equinox_](https://github.com/patrick-kidger/equinox). _Equinox_ allows for **differentiable classes** that are recognised as a valid _Jax_ type and _Zodiax_ adds lightweight methods to simplify interfacing with these classes! _Zodiax_ was originially built in the development of [dLux](https://github.com/LouisDesdoigts/dLux) and was designed to make working with large nested classes structures simple and flexible.

Zodiax is directly integrated with both Jax and Equinox, gaining all of their core features:

> - [Accelerated Numpy](https://jax.readthedocs.io/en/latest/jax-101/01-jax-basics.html): a Numpy like API that can run on GPU and TPU
>
> - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax-101/04-advanced-autodiff.html): Allows for optimisation and inference in extremely high dimensional spaces
>
> - [Just-In-Time Compilation](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html): Compliles code into XLA at runtime and optimising execution across hardware
>
> - [Automatic Vectorisation](https://jax.readthedocs.io/en/latest/jax-101/03-vectorization.html): Allows for simple parallelism across hardware and asynchronys execution
>
> - [Object Oriented Jax](https://docs.kidger.site/equinox/all-of-equinox/): Allows for differentiable classes that are recognised as a valid _Jax_ type
>
> - [Inbuilt Neural Networks](https://docs.kidger.site/equinox/api/nn/linear/): Has pre-built neural network layers classes
>
> - [Path-Based Pytree Interface](docs/usage.md): Path based indexing allows for easy interfacing with large and highly nested physical models
>
> - [Leaf Manipulation Methods](docs/usage.md): Inbuilt methods allow for easy manipulation of Pytrees mirroring the _Jax_ Array API

Doccumentataion: [louisdesdoigts.github.io/zodiax/](https://louisdesdoigts.github.io/zodiax/)

Installation: ```pip install zodiax```

Contributors: [Louis Desdoigts](https://github.com/LouisDesdoigts)

Requires: Python 3.8+, Jax 0.4.3+

---

### Quickstart

Create a regular class that inherits from `zodiax.Base`

```python
import jax
import zodiax as zdx
import jax.numpy as np

class Linear(zdx.Base):
    m : Jax.Array
    b : Jax.Array

    def __init__(self, m, b):
        self.m = m
        self.b = b

    def model(self, x):
        return self.m * x + self.b

linear = Linear(1., 1.)
```

Its that simple! The `linear` class is now a fully differentiable object that gives us **all** the benefits of jax with an object-oriented interface! Lets see how we can jit-compile and take gradients of this class.

```python
@jax.jit
@jax.grad
def loss_fn(model, xs, ys):
    return np.square(model.model(xs) - ys).sum()

xs = np.arange(5)
ys = 2*np.arange(5)
grads = loss_fn(linear, xs, ys)
print(grads)
print(grads.m, grads.b)
```

```python
> Linear(m=f32[], b=f32[])
> -40.0 -10.0
```

The `grads` object is an instance of the `Linear` class with the gradients of the parameters with respect to the loss function!

<!-- !!! tip "zodiax.filter_grad"
    If we replace the `jax.grad` decorator with `zdz.filter_grad` then we can choose speicifc parameters to take gradients with respect to! This is detailed in the [Using Zodiax section]((<https://louisdesdoigts.github.io/zodiax/docs/usage.md>)) of the docs.

!!! tip "Pretty-printing"
    All `zodiax` classes gain a pretty-printing method that will display the class instance in a nice readable format! Lets use it here to see what the gradients look like:

    ```python

    ```
    
    ```python
    > Linear(m=f32[], b=f32[])
    > -40.0 -10.0
    ``` -->
