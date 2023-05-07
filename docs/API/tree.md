# Tree

The Tree module provides a module for helpful pytree manipulation functions. It implements two functions, `boolean_filter(pytree, parameters)` and `set_array(pytree, parameters)`.

`boolean_filter(pytree, parameters)` returns a matching pytree with boolean leaves, where the leaves specified by `parameters` are `True` and the rest are `False`.

`set_array(pytree, parameters)` returns a matching pytree with the leaves specified by `parameters` set to the value of the corresponding leaf in `pytree`. This is to ensure they have a shape parameter in order to create dynamic array shapes for the bayesian module.

!!! info "Full API"
    ::: zodiax.tree