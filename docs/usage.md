
# All methods

`zodiax` also provies as series of extra methods designed to mirror those provided by the `jax.Array.at[]` [method](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html?highlight=.at):


- `.get(path)` - get the value of a leaf
- `.set(path, value)` - set the value of a leaf
- `.add(path, value)` - add a value to a leaf
- `.multiply(path, value)` - multiply a leaf by a value
- `.divide(path, value)` - divide a leaf by a value
- `.power(path, value)` - raise a leaf to a power
- `.min(path, value)` - take the minimum of a leaf and value
- `.max(path, value)` - take the maximum of a leaf and value
- `.apply(path, fn)` - applies the function to the leaf
- `.apply_args(path, fn, args)` - - `.apply(path, fn)` - applies the function to the leaf while also passing in the extra arguments

These methods are explored further in the (`Zodiax.Base` tutorial)[dont forget to add link]

On top of this there is the `Zodiax.ExtendedBase` class, that is designed to make working with the optimisation and inference libraries `Optax` and `Numpyro` much simpler!

 - `get_args(path)`
 - `get_param_spec(path, groups)`
 - `get_optimiser(path, optimisers)`
 - `update_and_model(model_fn, path, paths, values)`
 - `apply_and_model(model_fn, paths, fns)`

These methods are explored further in the [`Zodiax.ExtendedBase` tutorial](https://louisdesdoigts.github.io/zodiax/notebooks/ExtendedBase/)


---
# Nesting

`zodiax` also allows for paths and values to be nested, allowing for the updating of multiple leaves at once. Lets look at an example:

```python

from zodiax import Base

class Tree1(Base):
    a : object
    b : list
    c : dict

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

class Tree2(Base):
    a : float
    b : list
    c : dict

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

sub_tree = Tree2(1., [2, 3], {'k1': 4})
pytree = Tree1(sub_tree, [5, 6], {'k2': 7})
print(pytree)
```
```> Tree1(a=Tree2(a=1.0, b=[2, 3], c={'k1': 4}), b=[5, 6], c={'k2': 7})```

```python
new_pytee = pytree.set(['a.b', 'c'], [4]])
print(pytree)
```
```> Tree1(a=Tree2(a=1.0, b=10, c={'k1': 4}), b=10, c={'k2': 7})```


```python
new_pytee = pytree.set(['b', ['c', 'a.b.k1'], [4, 6]])
print(pytree)
```
```> Tree1(a=Tree2(a=0, b=-10, c=-10), b=10, c=10)```


This nesting works with all of the methods provided by `Zodiax`!

