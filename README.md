<!-- <h1 align='center'>Zodiax</h1> -->
# Zodiax

[![PyPI version](https://badge.fury.io/py/zodiax.svg)](https://badge.fury.io/py/zodiax)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![integration](https://github.com/LouisDesdoigts/zodiax/actions/workflows/tests.yml/badge.svg)](https://github.com/LouisDesdoigts/zodiax/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/LouisDesdoigts/zodiax/graph/badge.svg)](https://codecov.io/gh/LouisDesdoigts/zodiax)
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

Documentation: [louisdesdoigts.github.io/zodiax/](https://louisdesdoigts.github.io/zodiax/)

> Note: The Zodiax tutorials live in a [separate repo](https://github.com/LouisDesdoigts/zodiax_tutorials). This allows users to directly download and run the notebooks, ensuring that the correct packages needed to run them are installed! It also allows for new tutorials and examples to be added quite easily, without needing to update the core library.

Contributors: [Louis Desdoigts](https://github.com/LouisDesdoigts)

Requires: Python 3.10+, Jax 0.4.25+

Installation: 

```bash
pip install zodiax
```

Development installation: 

```bash
pip install "zodiax[dev]"
```

Coverage:

```bash
pytest --cov=zodiax --cov-report=term-missing --cov-report=xml --cov-report=html tests
```

This writes `coverage.xml` and an `htmlcov/` report for local inspection.

---

### Quickstart

Create a regular class that inherits from `zodiax.Base`

```python
import jax
import zodiax as zdx
import jax.numpy as np

class Linear(zdx.Base):
    m : jax.Array
    b : jax.Array

    def __init__(self, m, b):
        self.m = m
        self.b = b

    def __call__(self, x):
        return self.m * x + self.b

linear = Linear(1., 1.)
```

Its that simple! The `linear` class is now a fully differentiable object that gives us **all** the benefits of jax with an object-oriented interface! Lets see how we can jit-compile and take gradients of this class.

```python
@jax.jit
@jax.grad
def loss_fn(model, xs, ys):
    return np.square(model(xs) - ys).sum()

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

### Update Signatures (Minimal Overview)

Most Zodiax update methods (`set`, `add`, `multiply`, `divide`, `power`, `min`, `max`) support three equivalent input styles:

1. **`(parameters, values)` positional style**
2. **`{parameter: value}` dictionary style**
3. **`param=value` keyword style** (and `**{"nested.path": value}` for nested paths)

```python
# 1) Positional: (parameters, values)
linear = linear.set(["m", "b"], [2.0, 0.5])

# 2) Dictionary: {parameter: value}
linear = linear.add({"m": 0.1, "b": -0.2})

# 3) Keyword: param=value
linear = linear.multiply(m=2.0, b=0.5)
```

Use whichever style is clearest for your workflow. The operations remain immutable and return new objects.
