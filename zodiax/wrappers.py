import zodiax as zdx
import equinox as eqx
import jax
from jax import numpy as np
from jax.lax import dynamic_slice as lax_slice


def build_wrapper(eqx_model, filter_fn=eqx.is_array):
    arr_mask = jax.tree.map(lambda leaf: filter_fn(leaf), eqx_model)
    dyn, static = eqx.partition(eqx_model, arr_mask)
    leaves, tree_def = jax.tree.flatten(dyn)
    values = np.concatenate([val.flatten() for val in leaves])
    return values, EquinoxWrapper(static, leaves, tree_def)


class EquinoxWrapper(zdx.Base):
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


class WrapperHolder(zdx.Base):
    values: np.ndarray
    structure: EquinoxWrapper

    @property
    def build(self):
        return self.structure.inject(self.values)

    def __getattr__(self, name):
        if hasattr(self.structure, name):
            return getattr(self.structure, name)
        raise AttributeError(f"Attribute {name} not found in {self.__class__.__name__}")
