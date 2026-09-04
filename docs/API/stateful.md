# Arrays and state

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

## Downstream random expressions

Zodiax does not prescribe random distributions or key-stream semantics. A
downstream class can build directly on the expression protocol and ordinary JAX
random functions:

```python
import equinox as eqx
import jax.random as jr


class NormalNoise(zdx.Expression):
    shape: tuple[int, ...] = eqx.field(static=True)

    def evaluate(self, *, key, **context):
        return jr.normal(key, self.shape)


noise = NormalNoise((8,))
sample = noise.resolve(key=jr.key(0))
```

More specialised classes may store a `StateRef`, derive keys using `split` or
`fold_in`, or accept a key from call context. Those choices remain explicit and
natively interoperable with JAX. Built-in JAX random callables are registered with
Zodiax serialisation, and downstream callables can use `register_callable()`.

::: zodiax.numerics.arrays

::: zodiax.numerics.state
