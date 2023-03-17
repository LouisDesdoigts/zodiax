# Serialisation

!!! warning
    This module is experimental and is subject to change!

## Overview

This module is designed to be able to save and load models created in `zodiax` to and from a file. This is useful for saving the model for later use, or for sharing the model with others. Serialisation is a generally difficult problem to make fully robust, but can be constructed to cover most use cases!

There are two main types of functions in this module: the structure function and the serialisation functions. There are two structure functions:

1. `structure = build_structure(obj)`
2. `obj = load_structure(structure)`

The `build_structure` function traverses the input object and returns a `structure` dictionary that can be serialised. Each parameter in the object is either a 'container' node or a 'leaf' node, allowing the the full structure to be represented along with any nessecary meta-data required to reconstruct the object. The `load_structure()` function takes this `structure` dictionary and returns a pytree of the same structure that can be used in conjunction with `equinox.tree_serialise_leaves()` to return an identical object.

The serialisation functions are:

1. `serialise(obj, path)`
2. `obj = deserialise(path)`

The `serialise` function takes an object and a path, and saves the serialised object to the path. The `deserialise` function takes a path and returns the deserialised object.

---

## Future changes

There are some future improvements that are planned for this module, hence the present experimental status!

- [ ] Serialise package versions:

To try and ensure that the serialised object can be deserialised, the package versions should be serialised. This will allow the code to automatically check imported versions and raise warnings for imported package discrepancies.

- [ ] Add support for serialising functions:

This should also raise warning as functions can not in general be robustly serialised, but should be supported.

- [ ] Deal with static_fields:

There is a general issue with parameters in models that are marked as `equinox.static_field()`. Although this should rarely if even be used by the user, it is still a potential issue. Since the `equinox.tree_serialise_leaves()` function uses `tree_map` functions it is blind to these parameters. If this parameter is a string it is fine, however if it is some other data type it will at present not be serialised. This can be fixed by using the `tree_flatten()` function to determine what parameters are static and serialising them using a different method.

- [ ] Add support for serialising general objects:

In order to deal with the above `static_field()` issue, we must add support for serialising general python types, along wth array types.

- [ ] Implement robust tests:

The tests for this module are currently very basic, primarily becuase of the the tests are run in isolated enviroments, so classes that are created for the tests can not be re-imported.

When these changes have been implemented this module can be moved into the main `zodiax` package.

!!! info "Full API"
    ::: zodiax.experimental.serialisation