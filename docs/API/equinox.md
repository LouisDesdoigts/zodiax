# Equinox

Zodiax designed to be a 'drop in' replacement for Equinox, this means that all Equinox functions are available through Zodiax! Functions in the main Equinox namespace are raised into the Zodiax namespace, meaning these two line will import the *same* function:

```python
from equinox import filter_jit
from zodiax import filter_jit
```

Some Equinox functions are overwritten in order to give a path based interface. Currently there are two functions that are overwritten, `filter_grad` and `filter_value_and_grad`. This means that the following two lines will import *different* functions:

```python
from equinox import filter_grad
from zodiax import filter_grad
```

Submodules in Equinox are also raised into the Zodiax namespace through the `zodiax.equinox` submodule. This is how you would import the `nn` submodule from either Equinox or Zodiax:

```python
from equinox import nn
from zodiax.equinox import nn
```

!!! info "Full API"
    ::: zodiax.equinox