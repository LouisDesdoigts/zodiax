# 🌙 `Zodaix` ✨

[![PyPI version](https://badge.fury.io/py/zodiax.svg)](https://badge.fury.io/py/zodiax)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![integration](https://github.com/LouisDesdoigts/zodiax/actions/workflows/tests.yml/badge.svg)](https://github.com/LouisDesdoigts/zodiax/actions/workflows/tests.yml)
[![Documentation](https://github.com/LouisDesdoigts/zodiax/actions/workflows/documentation.yml/badge.svg)](https://louisdesdoigts.github.io/zodiax/)

Contributors: [Louis Desdoigts](https://github.com/LouisDesdoigts)


`Zodiax` is a `Jax` and `Equinox` based package designed to to simplify working with pytrees, geared towards their use in scientific programming! It allows users to build fully diffenetiable object-oriented software in `Jax` with a simple interface and extra functionality to make working with optimisation and inference libraries like `Optax` and `Numpyro` easy! If you are unfamiliar with `Jax` and `Equinox` we strongly reccomend reading the [Jax 101 tutorials](https://jax.readthedocs.io/en/latest/jax-101/index.html) and the [Equinox docs](https://docs.kidger.site/equinox/) to get an idea. 

`Zodaix` provides classes that extend the `equinox.Module` class, allowing for user-created classes to be treated as `Jax` objects! It is built on the principle that all classes are at a fundamental level a pytree and can be represented as a series of nested lists, tuples and dictionaries. Classes built using `Zodiax` gain the full power of `Jax` & `Equinox`, plus a series of methods that make working with and optimising complex nested class structures easy!

`Zodiax` was build during the development of [`dLux`](https://louisdesdoigts.github.io/dLux/), a fully differentiable optical modelling framework and spun out into its own package to aid others in building the next generation of powerful scientific programming tools that harness automatic-differentiation!

---

# 🐍 PyTrees 🌲

A pytree is a data structure used to represent hierarchical data. It is similar to a tree data structure, but instead of each node containing a single data element, each node in a pytree can have multiple children, allowing for more complex data structures to be represented. In python this is represented as a series of nested lists, tuples and dictionaries, and all classes are pytrees at the base level.

Lets look at some exampes:
    
```python
tree1 = (1, (2, 3), ())
tree2 = [1, {'k1': 2, 'k2': (3, 4)}, 5]
tree3 = {'a': 2, 'b': (2, 3)}
```

Each of the leaves of these pytrees have a unique path that can be used to access the value of the leaf. For example, the path to the value `3` in `tree1` is `(1, 1)` and the path to the value `4` in `tree2` is `(1, 'k2', 1)`. 

These ideas also apply to classes, lets look at some examples, using `Zodiax`:

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

The leaf `4` would then be accessed via the path `(a, c, 'k1')`, ie `pytree.a.c['k1']`.

Within the `Equinox` framework we need to declare the variables a class will hold as class attributes, and then assign them in the `__init__` method. This is because `Equinox` uses the `__init__` method to create a new instance of the class, and the class attributes are used to determine the structure of the class. Similarly becuase `Jax` objects are immutable, all classes we create are also immutable, which in short means that we can not perform in-place updates, but instead must create a new instance of the class with the updated values.

`Zodiax` extends this idea of paths, using strings to point to leaves within the class structure. The path to the value `4` in the `Zodiax` framework would then be `a.c.k1`. We can then use the `.get(path)` method!

```python
path = 'a.c.k1'
print(pytree.get('a.c.k1'))
```
```> 4```
 
These path objects then help to simplify the process of updating the values of leaves within a class. Within `Equinox` we would use the `equinox.tree_at(where, pytree, value)` method to update the value of a leaf, which we can still use within `Zodiax`! Say we want to update the value to 10, lets see how that would look:

```python
from equinox import tree_at
new_pytree = tree_at(lambda tree: tree.a.c['k1'], pytree, 10)
print(new_pytree)
```
```> Tree1(a=Tree2(a=1.0, b=[2, 3], c={'k1': 10}), b=[5, 6], c={'k2': 7})```


This is a bit of a mouthful, and we can simplify this using the `.set(path, value)` `Zodiax` method!

```python
new_pytree = pytree.set(path, 10)
print(new_pytree)
```
```> Tree1(a=Tree2(a=1.0, b=[2, 3], c={'k1': 10}), b=[5, 6], c={'k2': 7})```


---

# All methods

`Zodaix` also provies as series of extra methods designed to mirror those provided by the `jax.Array.at[]` [method](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html?highlight=.at):


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

These methods are explored further in the (`Zodiax.ExtendedBase` tutorial)[dont forget to add link]


---
# Nesting

`Zodaix` also allows for paths and values to be nested, allowing for the updating of multiple leaves at once. Lets look at an example:

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

---

# Extending `Zodaix`

## The `__getattr__` method

These `Zodaix` methods can be further extended using the `__getattr__` method, allowing users to create classes that contain a dictionary to have its keys accessed as attributes. This is done by defining a `__getattr__` method that checks if the attribute is in the dictionary, and if so returns the value. These can also be chained together in nested classes to reveal the leaves from deeply nested classes! Lets look at an example:

```python
from zodiax import Base

# Create a simple class with a single parameter
class SimpleTree(Base):
    value : float

    def __init__(self, value):
        self.value = value

simple_tree = SimpleTree(3.)

# Create a class with a dictionary of 'layers'
class LayersTree(Base):
    layers : dict
    parameter : float

    def __init__(self, layers, parameter):
        self.layers = layers
        self.parameter = parameter

    def __getattr__(self, key):
        """
        Search for the key in the dictionary.
        """
        if 'layers' in vars(self) and key in self.layers.keys():
            return self.layers[key]
        else:
            raise AttributeError(f'{key} not in {self.__class__.__name__}')

layers = {'layer1': 1.,
          'layer2': 2.,
          'layer3': 3.}
layer_tree = LayersTree(layers, 1.)

# Create a class that holds multiple classes
class ObjectTree(Base):
    object1 : object
    object2 : object

    def __init__(self, object1, object2):
        self.object1 = object1
        self.object2 = object2

    def __getattr__(self, key):
        """
        Itterate through the class atrributes and search for the key.
        """
        for attribute in vars(self).values():
            if hasattr(attribute, key):
                return getattr(attribute, key)
        raise AttributeError(f'{key} not in {self.__class__.__name__}')

object_tree = ObjectTree(layer_tree, simple_tree)

# Create a high level class
class MainTree(Base):
    obj : object

    def __init__(self, obj):
        self.obj = obj

    def __getattr__(self, key):
        for attribute in vars(self).values():
            if hasattr(attribute, key):
                return getattr(attribute, key)
        raise AttributeError(f'{key} not in {self.__class__.__name__}')

pytree = MainTree(object_tree)

# Examine the strucutre of the class
print(pytree)
```
```
> MainTree(
> obj=ObjectTree(
> object1=LayersTree(
>   layers={'layer1': 1.0, 'layer2': 2.0, 'layer3': 3.0},
>   parameter=1.0
>  ),
>  object2=SimpleTree(value=3.0)
>  )
> )
```


```python
# Access the values
print(pytree.parameter, pytree.layer2, pytree.value)
```
```> 1.0 2.0 3.0```

As we can see by implementing custom `__getattr__` methods we can access deeply nested leaves using the simple 'dot' notation! Not only this, but this work in conjunction with the paths used by `Zodiax`! Lets look at an example:

```python
new_pytree = pytree.set('layer2', 10)
```
```> 1.0 2.0 3.0```

This can massively simplify the process of updating the values of leaves within a class, and can be used to create a simple API for users to interact with the class.

### ✂️ Sharp Bits! ✂️

So there are a a few main things to be aware of using these methods. 

Firstly each leaf must have a unique name, by implementing the `__getattr__` method we are essentially reducing the path to that leaf to the name of that parameter or the dictionary key. If these classes have multiple leaves with the same name, then the first one found will be returned. This can be overcome with by being careful about how classes are structured and parameters are named!

Secondly the `hasattr` method in python does not work as one might initially expect with dictionaries! If we have a dictionary with a key of `key` and we call `hasattr(dict, 'key')` it will return `False`! This is because the `hasattr` method is actually calling `getattr(dict, 'key')` and checking if this raises an `AttributeError`. This is a bit of a gotcha, but can be overcome by checking if the key is in the dictionary keys, as shown in the `__getattr__` method in the `LayersTree` class above.

Thirdly any __getattr__ methods should raise an `AttributeError` if the key is not found, this is expected within python and ensures that these classes work correctly with the rest of python!

## The path_map (pmap) method

If we have a deeply nested class, we can define a 'path map' and pass it into the optional `pmap` keyword argument. This is simply a dictionary that allows for the keys to be used in place of the paths. Lets look at an example:

```python
pmap = {'param': 'a.c.k1'}
new_pytree = pytree.set('param', 10, pmap=pmap)
print(new_pytree)
```
```> Tree1(a=Tree2(a=1.0, b=[2, 3], c={'k1': 10}), b=[5, 6], c={'k2': 7})```

This can be usefull, but is ultimately made somewhat redundant by implementing a `__getattr__` method, however it can be a bit tricky, so this is a nice alternative!

---
