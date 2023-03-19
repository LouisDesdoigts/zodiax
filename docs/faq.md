# Frequently Asked Questions

`FrozenInstanceError: cannot assign to field ' '`

This error is common when trying to assign a value to a field in Zodiax. This isn't possible becuase Jax and therefore Zodiax is immutable. Imutable objects can't be modified in place, instead a new instance of the object with the value updated is returned. This is mitigated simply by using the `.set`, `.add`, `.multiply` etc methods:

```python
pytree = pytree.set('path', value)
pytree = pytree.add('path', value)
```

---

`ValueError: The following fields were not initialised during __init__: {' '}`

This error is common when trying to initialise an class that has a bad `__init__` method. In order to be registered as an immutable PyTree *all* parameters must be set in the constructor!