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

`Module` inherits `Base` and adds numerical-expression lifecycle methods plus two
conveniences for composite scientific models:

- `populate()` establishes shared `Linked` ownership;
- `resolve(**context)` evaluates ordinary `Expression` definitions while protecting
  runtime-declared fields;
- `realise(**context)` performs population and resolution together;
- `resolve(runtime=True, **context)` explicitly evaluates runtime fields;
- a uniquely named descendant can be raised to a shorter attribute lookup;
- an explicit local alias can map a public name to one real structural path.

```python
realised = model.realise(time=t)
```

A downstream owner can declare a context-sensitive field without changing its
expression type:

```python
class Stage(zdx.Module):
    response: object = zdx.field(runtime=True)
```

```python
class Model(zdx.Module):
    output: object


model = Model(
    output=zdx.Exp(x=zdx.Mul(x=zdx.Add(x=latent, b=offset), s=scale)),
    alias={
        "offset": "output.x.x.b",
        "latent": "output.x.x.x",
    },
)

model.offset
model.get("offset")
updated = model.set("offset", new_offset)
```

Alias input may be a mapping, one `(name, path)` pair, a sequence of pairs, or
`None`. It is stored as a sorted static tuple. Names must be unique public
identifiers and cannot collide with attributes declared on the owning module. An
alias may intentionally override or disambiguate a naturally raised descendant, and
several aliases may target the same leaf. Targets must be canonical dynamic
structural paths: they cannot cross static fields, properties, another alias, or a
raised shorthand.

Aliases are definition-time metadata. `populate`, `resolve`, or another subtree
replacement can remove the structure they target; lookup then reports a stale alias.
Use aliases on the unrealised model for configuration, optimisation, layouts, and
serialisation. `validate_aliases(tree)` checks a complete current topology.
Aliases are constructor inputs and static JAX tree metadata, so choose them outside
transformed loops.

The normal Equinox/Wadler-Lindig representation elides the dataclass default
`alias=None`. A non-`None` alias prints as a real field; Zodiax does not replace the
global Equinox `repr`.

Use `Base` for general path and archive behaviour. Use `Module` when numerical
expression lifecycle methods, raised descendants, or aliases are part of the model
API.

## Numerical expressions

`Expression` is the abstract protocol for a numerical definition evaluated later.
`Deferred` is specifically a key-only shared link, not the subclassing base:

```python
owner = zdx.Linked(value, key="shared")
reference = owner.defer()

mapped = zdx.Map(
    x=reference,
    s=scale,
    M=matrix,
    b=centre,
)

model = model.populate()  # inside the transformed calculation
model = model.resolve(time=time)
```

`Transform` objects use `x` for input and `y` for output. Parameters use the compact
semantic fields `b`, `s`, `M`, and `p`. The low-level functions are `add`, `mul`,
`power`, and `matmul`.

`Map` has the fixed order `y = matmul(s * x, M) + b`; `s`, `M`, and `b` are optional
identity stages. Nest ordinary transforms when their order differs; calling the
outer transform applies the complete stored spine and returns its realised array.

Optional numerical operands have a dataclass default of `None`. As with
`alias=None`, Equinox's ordinary representation omits them, so the printed object
shows the active expression rather than inactive placeholders.

Python scalars at numerical constructor boundaries become strong inexact arrays.
The public `as_array` helper preserves inferred dtype and passes through `None` or
any `Expression`.

## Global numerical units

`set_units(mapping)` selects one coherent realised unit per numerical category for
subsequently constructed `Unit` transforms:

```python
zdx.set_units({"cartesian": "um", "angular": "mas"})

distance = zdx.Unit(2.0, "mm")
distance.resolve()  # 2000.0 um
```

The transform stores and prints both its input `unit` and captured output `to` unit.
It remains an ordinary reversible `Transform`; realised values are JAX arrays, not
quantity wrappers.

## State

`State` owns named running values and uses the ordinary Zodiax path API. `StateRef`
contains only a static path into the State supplied during resolution:

```python
state = zdx.State(
    index=0,
    time=0.0,
    dt=0.1,
    key=jr.key(0),
)
scheduled = zdx.Mul(x=state.ref("time"), s=2.0)

current = scheduled.resolve(state=state)
state = state.step()
zdx.validate_state(scheduled, state)
```

State remains an explicit input and output of an ordered model step. Any stored key
is left unchanged by `step()`; downstream expressions define how ordinary JAX keys
are derived and consumed.

## Tree definitions

The sibling `zodiax.derivatives` package provides `TreeLayout` and the associated
realised vector, matrix, Jacobian, exact Hessian, Gauss--Newton, and Fisher classes.
`TreeLayout` records selected floating-coordinate paths and shapes. Derivative
operations include every leaf of the supplied parameter PyTree and reject
non-floating leaves rather than silently filtering them. `ObjectDefinition`
separately records the complete supported object topology for validated `.zdx`
archives and is generated automatically by `save` and `load`.
