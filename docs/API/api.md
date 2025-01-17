# Modules Overview

Zodiax is a lightweight extension of Equinox and so only has a few modules and methods. Lets check them out!

---

# Base

The `Base` class is the foundational object of Zodiax and is what allows for a path-based pytree interface. Classes that inherit from `Base` will gain methods that allow for operations and functions to be applied to leaves specified by their paths. Here is a summary of the methods:

**Getter Methods**

```python
value = pytree.get(paths)
```

**Setter Methods**

```python
pytree = pytree.set(paths, values)
# pytree = pytree.set_and_call(paths, values, call_fn)
pytree = pytree.update(dict)
```

**Arithmetic Methods**

```python
pytree = pytree.add(paths, values)
pytree = pytree.multiply(paths, values)
pytree = pytree.divide(paths, values)
pytree = pytree.power(paths, values)
pytree = pytree.min(paths, values)
pytree = pytree.max(paths, values)
```

**Functional Methods**

```python
pytree = pytree.apply(paths, fns)
pytree = pytree.apply_args(paths, fns, args)
pytree = pytree.apply_and_call(paths, fns, call_fn)
```

---

# Equinox

Zodiax designed to be a 'drop in' replacement for Equinox, this means that all Equinox functions are available through Zodiax! Functions in the main Equinox namespace are raised into the Zodiax namespace, meaning these two line will import the *same* function:

```python
from equinox import filter_jit
from zodiax import filter_jit
```

Some Equinox functions are overwritten in order to give a path based interface. Currently there are three functions that are overwritten: `filter_grad`, `filter_value_and_grad`, and `partition`. This means that the following two lines will import *different* functions:

```python
from equinox import filter_grad
from zodiax import filter_grad
```

Submodules in Equinox are also raised into the Zodiax namespace through the `zodiax.equinox` submodule. This is how you would import the `nn` submodule from either Equinox or Zodiax:

```python
from equinox import nn
from zodiax.equinox import nn
```

<!-- ---

# Optimisation

The `zodiax.optimisation` module contains only a single function, `get_optmiser`. It is a simple interface designed to apply Optax optimisers to individual leaves!

---

# Tree

The Tree module provides a module for helpful pytree manipulation functions. It only implements a single function, `get_args(paths)`. It returns a matching pytree with boolean leaves, where the leaves specified by `paths` are `True` and the rest are `False`.

---

# Serialisation

!!! warning "Serialisation is currently an experimental Module"
    This module is currently experimental and may change in future versions.

The Serialisation methods are designed to make it easy to save and load Zodiax models! There are two main functions: `serialise()` and `deserialise()`. -->