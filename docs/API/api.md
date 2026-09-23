# Modules overview

Zodiax is a lightweight extension of Equinox. Its APIs operate on immutable PyTrees
and use dotted paths for leaf selection.

## `Base`

`Base` extends `equinox.Module` with declared-path operations:

```python
value = model.get(paths)

model = model.set(paths, values)
model = model.add(paths, values)
model = model.multiply(paths, values)
model = model.divide(paths, values)
model = model.power(paths, values)
model = model.min(paths, values)
model = model.max(paths, values)
```

The same class exposes general archive helpers:

```python
model.save("model.zdx")
loaded = model.load("model.zdx")
```

## `Module` and aliases

Module adds model preparation and concise parameter lookup to Base's path operations.

Its implementation lives in `zodiax.module`; the public names remain `zdx.Module`,
`zdx.Alias`, `zdx.update`, and `zdx.validate_aliases`.

```python
import jax
import jax.numpy as np
import zodiax as zdx


class Surface(zdx.Module):
    coefficients: jax.Array | zdx.Expression


class Optic(zdx.Module):
    surface: Surface


optic = Optic(
    surface=Surface(coefficients=np.array([0.1, 0.2])),
    alias={"amplitudes": "surface.coefficients"},
)
coefficients = optic.coefficients  # Raised from the nearest matching descendant.
same = optic.amplitudes           # Selected by a local structural alias.
updated = optic.set(amplitudes=np.array([0.3, 0.4]))
```

Declared attributes take precedence, followed by local aliases and the nearest
matching descendants. Equal-depth ambiguity requires a qualified path. Alias input
is stored as a canonical static tuple of `(name, path)` pairs, with unique public
names and targets following dynamic structural fields. Aliases do not add leaves.

`resolve(**context)` populates shared links and prepares available Expression fields,
returning the same ordinary Module class. `populate()` performs only structural link
replacement. Resolving a nested definition can remove an alias target; keep edits on
the definition and use `validate_aliases()` to check a changed topology explicitly.
Inherited `set` does not rerun constructor validation.

`zdx.update(parameters, *models)` returns models in their original order. Default
`mode="first"` assigns each path to its first match; `mode="all"` updates every match.
`strict=True` rejects paths unused by every supplied object.

The [Module reference](module.md) gives complete examples and typed contracts for
aliases, raised lookup, immutable updates, and prepared models.

## Numerical expressions

`Expression` is the abstract protocol for a numerical definition evaluated later.
`Deferred` is specifically a key-only shared link, not the subclassing base:

```python
import jax
import zodiax as zdx


class Measurements(zdx.Module):
    science: jax.Array | zdx.Expression
    reference: jax.Array | zdx.Expression


gain = zdx.Linked(2.0, key="gain")
model = Measurements(
    science=zdx.Map(x=3.0, s=gain),
    reference=zdx.Map(x=1.0, s=gain.defer()),
)
prepared = model.resolve()  # Inside the differentiated calculation.
print(model)
print(prepared)
print(prepared.science, prepared.reference)
```

```text
Measurements(
  science=Map(x=f32[], s=Linked(value=f32[], key='gain')),
  reference=Map(x=f32[], s=Deferred(key='gain'))
)
Measurements(science=f32[], reference=f32[])
6.0 2.0
```

`Transform` objects use `x` for input and `y` for output. Map parameters use `b`,
`s`, and `M`; the public `matmul` function provides matrix or basis contraction.
`Operation` subclasses Exp, Log, Pow, and Unit use ordinary JAX broadcasting for
their elementwise formulas.

`Map` has the fixed order `y = matmul(s * x, M) + b`; `s`, `M`, and `b` are optional
identity stages. Nest ordinary transforms when their order differs; calling the
outer transform with an explicit input applies the complete nested Transform chain.
An empty call resolves its stored definition. `apply(value, inverse=True)` and
`transform(value, inverse=True)` reverse the chain using each transform's local
`inv`; Map and Mask provide representative reverses. Local `fwd` and `inv` operate
on prepared arrays. The advanced `initialise(y)` helper binds a recovered deepest
input in a new definition.

Automatic link population covers the tree supplied to `resolve`. Resolve the
containing Module or complete unbound operator before explicit calls involving
references across operands. See the [numerics reference](numerics.md) for the
class contracts and [design](../numerics_design.md) for the full hierarchy.

Optional numerical operands have a dataclass default of `None`. As with
`alias=None`, Equinox's ordinary representation omits them, so the printed object
shows the active expression rather than inactive placeholders.

Python scalars at numerical constructor boundaries become strongly typed arrays
with their inferred numerical type. The public `as_array` helper can request
`float`, `int`, `complex`, or `bool` using JAX's configured default precision.
`None` and Expressions pass through unchanged without attaching a future cast.

## Global numerical units

`set_units(mapping)` selects the output convention used by Unit transforms whose
`unit_out` is `None`:

```python
import zodiax as zdx

previous_units = zdx.get_units()
zdx.set_units({"cartesian": "um", "angular": "mas"})

distance = zdx.Unit(x=2.0, unit="mm")
print(distance)
print(distance.resolve())
zdx.set_units(previous_units)
```

```text
Unit(x=f32[], unit='mm')
2000.0
```

The transform stores its input `unit` and optional `unit_out`. With the default
`None`, eager evaluation follows current global settings. An explicit `to="m"`
in the constructor fixes the output. `distance.to(unit, **context)` returns the
converted array without changing the definition; `distance.to(None)` converts into
the current global convention. An absent input or missing context raises
`ValueError`. JIT captures the convention when tracing, so changing globals does
not alter an already compiled call. Unit is an elementwise Operation with an inverse;
realised values remain ordinary arrays.

## State

`State` owns named running values and uses the ordinary Zodiax path API. `StateRef`
contains only a static path into the State supplied during resolution:

```python
import jax.random as jr
import zodiax as zdx

state = zdx.State(
    index=0,
    time=0.0,
    dt=0.1,
    key=jr.key(0),
)
scheduled = zdx.Map(x=state.ref("time"), s=2.0)

current = scheduled.resolve(state=state)
state = state.step()
zdx.validate_state(scheduled, state)
```

State remains an explicit input and output of an ordered model step. Any stored key
is left unchanged by `step()`; downstream expressions define how ordinary JAX keys
are derived and consumed.

## Tree definitions

The sibling `zodiax.derivatives` package provides realised vector, matrix, Jacobian,
exact Hessian, Gauss--Newton, and Fisher classes, plus reusable eigendecomposition,
Cholesky, projection, and local-parameterisation results. Derivative operations
derive backend coordinate tracking from the supplied parameter PyTree automatically;
`TreeLayout` is the lower-level metadata object that records those paths and shapes.
Every supplied leaf is included and non-floating leaves are rejected rather than
silently filtered. `ObjectDefinition` separately records the complete supported
object topology for validated `.zdx` archives and is generated automatically by
`save` and `load`.
