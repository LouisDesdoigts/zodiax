# Extending `zodiax`

## The `__getattr__` method

These `zodiax` methods can be further extended using the `__getattr__` method, allowing users to create classes that contain a dictionary to have its keys accessed as attributes. This is done by defining a `__getattr__` method that checks if the attribute is in the dictionary, and if so returns the value. These can also be chained together in nested classes to reveal the leaves from deeply nested classes! Lets look at an example:

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