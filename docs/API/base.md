# Base

Base provides immutable dotted-path operations on an Equinox PyTree.

```python
import jax.numpy as np
import zodiax as zdx


class Parameters(zdx.Base):
    values: dict


parameters = Parameters(values={"gain": np.array(2.0), "offset": np.array(1.0)})
updated = parameters.set("values.gain", np.array(3.0))
assert np.allclose(parameters.get("values.gain"), 2.0)
assert np.allclose(updated.get("values.gain"), 3.0)
```

`get` reads declared paths; `set` returns a copy with selected values replaced.
Arithmetic path methods such as `add`, `multiply`, and `divide` apply an operation
at the selected leaves and return a new object. Base retains its general container
and metadata behaviour. Replacement values are supplied by the caller; inherited
`set` does not rerun constructors.

Use [Module](module.md) for composite numerical models that need aliases, nearest
named descendants, shared-value population, or expression resolution. Module
inherits these Base path operations.

`Base.save(path)` writes a validated Zodiax archive. Calling `template.load(path)`
loads through the same format while using `template` to verify the expected class and
field structure.

::: zodiax.base
