# Base

The `Base` class is the foundational object of Zodiax and is what allows for a path-based pytree interface. Classes that inherit from `Base` will gain methods that allow for operations and functions to be applied to leaves specified by their paths. Here is a summary of the methods:

**Getter Methods**

```python
value = pytree.get(paths)
```

**Setter Methods**

```python
pytree = pytree.set(paths, values)
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

!!! info "Full API"
    ::: zodiax.base.Base