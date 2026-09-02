# Arrays, state, and random keys

Numerical constructor boundaries can use `as_array` to convert Python or NumPy
values to strongly typed JAX arrays while preserving inferred dtype. `None` and any
`Expression` pass through unchanged.

## Running state

`State` is an immutable Zodiax `Module` that owns named array or expression values.
It uses the same path-based `get` and `set` API as every other Zodiax module:

```python
state = zdx.State(time=0.0, counter=0, key=jr.key(0))

state.time
state.get("time")
state = state.set(time=1.5, counter=1)
```

Numerical inputs are converted to arrays at construction and after updates. The
state topology and every leaf's shape, dtype, and weak type remain fixed; `set`
replaces existing paths rather than adding new entries. This is the carry contract
required by JAX compiled loops. Nested mappings provide namespaced state when
needed.

A conventional State can advance its available index and time entries in one pure
transition:

```python
state = zdx.State(index=0, time=0.0, dt=0.1, key=jr.key(0))

state = state.step()
```

`step()` increments `index` when present and adds the stored `dt` to `time` when
present. `step(dt=...)` overrides the stored time step for one transition. A `key`
entry remains a stable root rather than becoming a chained sequence of keys.

`StateRef(path)` is an `Expression` containing only a static path. `State.ref(path)`
constructs the same reference while checking immediately that the path exists:

```python
time = state.ref("time")
same_time = zdx.StateRef("time")

current_time = time.resolve(state=state)
same_value = same_time.resolve(state=state)
```

References compose anywhere an expression is accepted. Interpolation therefore
uses its ordinary `x` operand for state-driven queries:

```python
history = zdx.Interpolation(
    knots=times,
    values=samples,
    x=state.ref("time"),
)

value = history.resolve(state=state)
state = state.step()
next_value = history.resolve(state=state)
```

`validate_state(model, state)` checks that every reference targets an available
path. Several references may intentionally share one state entry, and unrelated
state entries are permitted.

## Random expressions

`Random` wraps any JAX-style key-first callable without requiring one Zodiax class
per distribution:

```python
noise = zdx.Random(
    jr.normal,
    shape=(8,),
    stream="noise",
)

sample = noise.resolve(state=state)
state = state.step()
next_sample = noise.resolve(state=state)
```

The expression uses ordinary `jax.random.fold_in` operations to derive a key from
`state.key`, `state.index`, and its static stream identity. The root key does not
change. An explicit `key=` and `index=` can instead make a Random definition
self-contained. Without any index, index zero gives a fixed reproducible
realisation. Common JAX trace-time options such as `shape`, `dtype`, and `axis` are
static automatically; custom named options can be selected with `static=`.

Common JAX random functions are registered for Zodiax serialisation. Custom
key-first callables use the normal `register_callable()` mechanism.

::: zodiax.numerics.arrays

::: zodiax.numerics.state

::: zodiax.numerics.random
