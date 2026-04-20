<!-- <h1 align='center'>Zodiax</h1> -->
# Zodiax

[![PyPI version](https://badge.fury.io/py/zodiax.svg)](https://badge.fury.io/py/zodiax)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![integration](https://github.com/LouisDesdoigts/zodiax/actions/workflows/tests.yml/badge.svg)](https://github.com/LouisDesdoigts/zodiax/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/LouisDesdoigts/zodiax/graph/badge.svg)](https://codecov.io/gh/LouisDesdoigts/zodiax)
[![Documentation](https://github.com/LouisDesdoigts/zodiax/actions/workflows/documentation.yml/badge.svg)](https://louisdesdoigts.github.io/zodiax/)

---

[_Zodiax_](https://github.com/LouisDesdoigts/zodiax) is an differentiable object-oriented framework geared towards scientific programming and physical modelling. Its built on the [_JAX_](https://github.com/google/jax) + [_Equinox_](https://github.com/patrick-kidger/equinox) ecosystem, inherits all of their functionality, and adds a series of extra methods to make working with physically representative classes simple and flexible. On top of that, it also adds a number of helpful optimisation and statistics tools often found in the physical sciences but not in the machine learning field. Zodiax was spun out from the development of [dLux](https://github.com/LouisDesdoigts/dLux), a differentiable optics framework that still uses Zodiax as its core class structure.

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

Its that simple! The `linear` class is now a fully differentiable object that gives us **all** the benefits of jax with an object-oriented interface!

### Manipulating leaves

Zodiax provides a number of methods to manipulate the leaves of a class. These methods allow you to update, add, multiply, divide, and perform other operations on the parameters of a class in a flexible and intuitive way. These update methods support three equivalent input styles:

1. **`(parameters, values)` positional style**
2. **`{parameter: value}` dictionary style**
3. **`param=value` keyword style**

```python
# 1) Positional: (parameters, values)
linear = linear.set(["m", "b"], [2.0, 0.5])

# 2) Dictionary: {parameter: value}
linear = linear.add({"m": 0.1, "b": -0.2})

# 3) Keyword: param=value
linear = linear.multiply(m=2.0, b=0.5)
```

Use whichever style is clearest for your workflow. The operations remain immutable and return new objects.

### Next steps

There are two main tutorials for Zodiax. The first is the [Building Classes](https://louisdesdoigts.github.io/zodiax/usage/) tutorial, which give s a detailed overview of how to construct classes, how neasted object and paths work, and how to manipulate them. The second is the [Optimisation and Inference](https://louisdesdoigts.github.io/zodiax/optimisation_tools/) tutorial, which gives an relatively complete overview of the how to perform the most common optimisation and inference problems using Zodiax objects, and introduces a lot of the various tools and methods provided by the repo. If you are familiar with both of those, then you pretty much know all of Zodiax!