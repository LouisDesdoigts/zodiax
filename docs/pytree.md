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
