
# Using Zodiax

## Resources

_Zodiax_ is built from both _Jax_ and _Equinox_, so if you are unfamiliar with those packages you should go through their docs and tutorials first! Here are some resources to to get you started: 

- [Jax 101 tutorials](https://jax.readthedocs.io/en/latest/jax-101/index.html)
- [Jax Pytrees](https://jax.readthedocs.io/en/latest/pytrees.html)
- [All of Equinox](https://docs.kidger.site/equinox/all-of-equinox/)

## Constructing Zodiax classes

`zodiax.Base` is the core class of Zodiax that registers instances of the class as a Pytree which is a native Jax object. Zodiax is also designed to ease working with complex nested class structures often nesseciateted by large physical models common to scientific programming. To do this `zodiax.Base` has a series of class methods that reflect the `jax` array update methods, along with introducing the concept of paths that can be used to access and update the leaves of a pytree.

Lets examine how these methods work by looking at an example class structure. We will start with a simple class that models a normal distribution and then build a class that contains multiple instances of this class:

```python
import zodiax as zdx
from jax import numpy as np, scipy as scp

class Normal(zdx.Base):
    mean      : np.ndarray
    scale     : np.ndarray
    amplitude : np.ndarray

    def __init__(self, mean, scale, amplitude):
        self.mean      = np.asarray(mean,      dtype=float)
        self.scale     = np.asarray(scale,     dtype=float)
        self.amplitude = np.asarray(amplitude, dtype=float)
    
    def model(self, width=10):
        xs = np.linspace(-width, width, 128)
        return self.amplitude * scp.stats.norm.pdf(xs, self.mean, self.scale)
```

This class simply models a normal distribution with a mean, scale and amplitude, and has a `.model()` method that is used to actually perform the calculation of the normal distribution.

!!! tip "Declaring attributes"
    When using `equinox` or `zodiax` the attibutes of the class must be
    declared in the class definition to determine the structure of the
    pytree that is created when the class is instantiated. This is done by
    adding a type hint to the attribute which can be any valid python type
    and is **not** type checked!

!!! info "`.model()` vs `.__call__()`"
    It is common in Equinox to not define a `.model()` method but rather a `.__call__()` method so that the instance of the class can be called like a function, ie:

    ```python
    normal = Normal(0, 1, 1)
    distribution = normal(10)
    ```

    This is a matter of personal preference, *however* when using Optax if you try to optimise a class that has a `.__call__()` method, you can thrown unhelpful errors. Becuase of this I recommend avoiding `.__call__()` methods and instead using `.model()` method.

Now we construct a class to store and model a set of multiple normals.

```python
class NormalSet(zdx.Base):
    normals : dict
    width   : np.ndarray

    def __init__(self, means, scales, amplitude, names, width=10):
        normals = {}
        for i in range(len(names)):
            normals[names[i]] = Normal(means[i], scales[i], amplitude[i])
        self.normals = normals
        self.width = np.asarray(width, dtype=float)
    
    def __getattr__(self, key):
        if key in self.normals.keys():
            return self.normals[key]
        else:
            raise AttributeError(f"{key} not in {self.normals.keys()}")
    
    def model(self):
        return np.array([normal.model(self.width) 
            for normal in self.normals.values()]).sum(0)

sources = NormalSet([-1., 2.], [1., 2.], [2., 4.], ['alpha', 'beta'])
```

The `Normal` class is a simple class that represents a normal distributions. The `NormalSet` class is a container for multiple `Normal` objects. The `NormalSet` class also has a `model` method that returns the sum of the models of each of the `Normal` objects. This is a simple example of a nested class structure, where the `NormalSet` class contains multiple `Normal` objects.

!!! question "Whats with the `__getattr__` method?"
    This method eases working with nested structures and canbe used to raise parameters from the lowst level of the class structure up to the top. In this example it allows us to access the individual `Normal` objects by their dictionary key. Using this method, these two lines are equivalent:

    ```python
    mu = sources.normals['alpha'].mean
    mu = sources.alpha.mean
    ```

    These methods can be chained together with multiple nested classes to make accessing parameters across large models much simpler!

    It is strongly reccomended that your classes have a `__getattr__` method implemented as it makes working with nested structures *much* easier! When doing so it is important to ensure that the method raises the correct error when the attribute is not found. This is done by raising an `AttributeError` with a message that includes the name of the attribute that was not found. 

Lets print this object to have a look at what it looks like:

```python
print(source)
```

```python
> NormalSet(
>   normals={
>     'alpha':
>     Normal(mean=f32[], scale=f32[], amplitude=f32[]),
>     'beta':
>     Normal(mean=f32[], scale=f32[], amplitude=f32[])
>   },
>   width=f32[]
> )
```

!!! question "Whats with the f32[2]?"
    The `f32[2]` is the `jax` representation of a `numpy` array. The `f32` is the dtype and the `[2]` is the shape. The `jax` representation of a scalar is `f32[]`.

## Working with Zodiax classes

### **Paths**

Paths are a simple concept that allows us to index a particular leaf of the pytree. The path is a string that is constructed by concatenating the names of the attributes that lead to the leaf. Regardless of the data type of the node, the path is always a string and joined by a period '.', here are some paths for the `source` class instance:

```python
"normals.alpha.mean"
"normals.alpha.scale"
"normals.beta.amplitude"
```

Since we have constructed the `__getattr__` method, these paths can be simplified to:

```python
"alpha.mean"
"alpha.scale"
"beta.amplitude"
```

!!! tip "Path Uniqueness"
    Paths must be unique
    Paths should not have space in them to work properly with the `__getattrr__`

### **Added Class Methods**

Zodiax adds a series of methods that can be used to manipulate the nodes or leaves of these pytrees that mirror and expand the functionality of the `jax.Array.at[]` [method](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html?highlight=.at). The main methods are `get`, `set`, `add`, `multiply`, `divide`, `power`, `min`, `max`, `apply` and `apply_args`. The first arugument to all of these functions is a path and those that manipulate leaves also take in a leaf parameter! update They all essentially follow the same syntax so lets look at some examples of how we would perform basic operations to Zodiax obejcts.

Lets change our 'alpha` source to a unit normal:

```python
sources = sources.set('alpha.mean', 0.)
sources = sources.set('alpha.scale', 1.)
sources = sources.set('alpha.amplitude', 1.)
print(sources.alpha)
```

```python
> Normal(mean=0.0, scale=1.0, amplitude=1.0)
```

!!! question "Wait where did the `f32[]` go?"
    This is because we have replaced the `jax` array with a python float!. It is important to note that the `set` method does not perform any type checking and will simply replace the leaf with whatever is passed in. Be careful when setting leaves to make sure they are the correct type and that you dont get unexpected errors down the line!

!!! warning "Immutability"
    Since Jax is immutable, Zodiax is also immutable. All this means is we can not update values in place and instead create a new instance of an object with the updated value.

    In regular (mutable) python if we wanted to update the value of some parameter in a class we would do something like this:

    ```python
    sources.alpha.mean = 4
    sources.alpha.mean += 2
    ```

    However in Zodiax this will throw a `FrozenInstanceError`, what gives! Lets see how we can use Zodiax to achieve the same thing:

    ```python
    sources = sources.set('alpha.mean', 4)
    sources = sources.add('alpha.mean', 2)
    ```

### **Multiple Paths and Nesting**

Zodiax in very felixible in how you can use the paths to access and manipulate the leaves of the pytree. You can use a single path to access a single leaf, or you can use a list of paths to access multiple leaves. You can also use nested paths to access nested leaves. Lets see some examples:

Lets add all of the paths to the means together so we can have a simple variable that we can use to globally shift all of the `sources` at the same time.

```python
means = ['alpha.mean', 'beta.mean', 'gamma.mean']
shifted_sources = sources.add(means, 2.5)
```

It's that easy! We can also nest paths in order to perform complex operations simply. Lets say we want to change the scale of both the 'alpha' and 'beta' source together and the 'gamma' source independently.

```python
scales = [['alpha.scale', 'beta.scale'], 'gamma.scale']
values = [2., 4.]
scaled_sources = sources.multiply(scales, values)
```

This concept apllies to **all** of Zodiax and can be used with any of its methods. Similarly Zodiax is designed so that every update operations is performed **simultaneously**. This prevents the unessecary overhead of copy the entire contents of the pytree for every update which is especcially beneficial for large models!

<!-- ## Optimisation and Inference in Zodiax

This tutorial will cover how to use Zodiax to perform optimisation and inference on models. We will start with a simple example of optimising a model using gradient descent, then show how to use the good deep mind gradient processing library [Optax](https://optax.readthedocs.io/en/latest/), then show how to use numpy to perform inference on the data, and finally show how to use derivates to calcaulte Fisher matrices. Lets use the classes we created above -->

