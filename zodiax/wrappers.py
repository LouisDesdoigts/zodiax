import zodiax
import equinox as eqx
import jax
from jax import numpy as np
from jax.lax import dynamic_slice as lax_slice


__all__ = ["build_wrapper", "EquinoxWrapper", "WrapperHolder"]


def build_wrapper(eqx_model, filter_fn=eqx.is_array):
    """
    Deconstructs an equinox model into its values and structure, and returns a
    `WrapperHolder` object that can be used to interact with the model in a way
    that is compatible with the Zodiax framework.

    Parameters
    ----------
    eqx_model : equinox.Module
        The Equinox model to deconstruct.
    filter_fn : callable, optional
        A function that takes a leaf of the model and returns a boolean value

    Returns
    -------
    values : np.ndarray
        The values of the model, flattened and concatenated.
    structure : EquinoxWrapper
        The structure of the model, stored in a `EquinoxWrapper` object.
    """
    arr_mask = jax.tree.map(lambda leaf: filter_fn(leaf), eqx_model)
    dyn, static = eqx.partition(eqx_model, arr_mask)
    leaves, tree_def = jax.tree.flatten(dyn)
    values = np.concatenate([val.flatten() for val in leaves])
    return values, EquinoxWrapper(static, leaves, tree_def)


class EquinoxWrapper(zodiax.base.Base):
    """
    A wrapper class designed to store an Equinox model (typically a neural network)
    in a way that makes it easily compatible within the Zodiax framework. This is
    necessary as Equinox operates on _whole_ models, where as Zodiax operates on
    model _leaves_. This class is designed to bridge that gap.

    This class should not need to be interacted with directly, and is designed to be
    held within the `WrapperHolder` class.
    """

    static: eqx.Module
    shapes: list
    sizes: list
    starts: list
    tree_def: None

    def __init__(self, static, leaves, tree_def):
        self.static = static
        self.tree_def = tree_def
        self.shapes = [v.shape for v in leaves]
        self.sizes = [int(v.size) for v in leaves]
        self.starts = [int(i) for i in np.cumsum(np.array([0] + self.sizes))]

    def inject(self, values):
        leaves = [
            lax_slice(values, (start,), (size,)).reshape(shape)
            for start, size, shape in zip(self.starts, self.sizes, self.shapes)
        ]
        return eqx.combine(jax.tree.unflatten(self.tree_def, leaves), self.static)


class WrapperHolder(zodiax.base.Base):
    """
    A class designed to hold an Equinox model, its structure and values. This helps it
    operate smoothly within the Zodiax framework.

    To apply transformations to the Equinox model values, operate on the `values` leaf
    of this class. To build the model, call the `build` property, and the Equinox model
    will be constructed with the stored values and be able to operated with as if it
    were a regular Equinox model.

    This class is designed to be instantiated by the `build_wrapper` function.

    Example
    -------

    ```python
    import equinox as eqx
    import zodiax as zdx
    import jax.numpy as np
    import jax.random as jr

    eqx_model = eqx.nn.MLP(
        in_size=16, out_size=16, width_size=32, depth=1, key=jr.PRNGKey(0)
    )

    class Foo(zdx.WrapperHolder):

        def __init__(self, nn):
            values, structure = zdx.build_wrapper(nn)
            self.values = values
            self.structure = structure

        def __call__(self, x):
            return self.build(x)

    x = np.ones(16)
    foo = Foo(eqx_model)

    # Now we can use the model as if it were a regular Equinox model
    print(foo(x))

    >>> [ 0.1767296   0.15628047 -0.63250038 -0.01583058  0.39692974  0.4556041
    >>>   0.33121592 -0.3183221  -0.75008567 -0.32724514  0.28351735 -0.03595607
    >>>  -0.53921278 -0.20966474 -0.33641739 -0.28726151]

    # We can also apply Zodiax transformations to the model!
    print(foo.multiply("values", 0.)(x))
    >>> [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    """

    values: np.ndarray
    structure: EquinoxWrapper

    @property
    def build(self):
        """
        Builds the Equinox model with the stored values and structure.
        """
        return self.structure.inject(self.values)

    def __getattr__(self, name):
        if hasattr(self.structure, name):
            return getattr(self.structure, name)
        raise AttributeError(f"Attribute {name} not found in {self.__class__.__name__}")
