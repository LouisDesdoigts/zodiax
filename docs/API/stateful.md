# Arrays and state

Numerical constructor boundaries can use `as_array` to convert Python or NumPy
values to strongly typed JAX arrays while preserving inferred dtype. `None` and any
`Expression` pass through unchanged.

Generic conversion retains inferred integer and complex types. Passing
`dtype=float`, `int`, `complex`, or `bool` requests that numerical kind using JAX's
configured default precision. The representations below use the default
configuration.

## Running state

`State` is an immutable Zodiax `Module` of current numerical arrays. Expressions
belong in the model; StateRef reads their current inputs from state.
It uses the same path-based `get` and `set` API as every other Zodiax module:

```python
import jax.random as jr
import zodiax as zdx

state = zdx.State(time=0.0, counter=0, key=jr.key(0))

state.time
state.get("time")
state = state.set(time=zdx.as_array(1.5), counter=zdx.as_array(1))
print(state.time, state.counter)
```

```text
1.5 1
```

Construction converts each entry to one array, including numerical lists and
tuples and typed JAX random keys. State entries must be numerical arrays; `None`
and Expression definitions are not valid entries. Inherited `set` does not
normalise replacements; supply arrays. When State is a JAX loop carry, callers
preserve its tree structure and every array's shape and dtype. Qualified paths
such as `values.time` disambiguate entry names.

A conventional State can advance its available index and time entries in one pure
transition:

```python
state = zdx.State(index=0, time=0.0, dt=0.1, key=jr.key(0))
print(state)
state = state.step()
print(state.index, state.time)
```

```text
State(values={'index': i32[], 'time': f32[], 'dt': f32[], 'key': key<fry>[]})
1 0.1
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
print(time)
print(current_time, same_value)
```

```text
StateRef(path='time')
0.1 0.1
```

References compose anywhere an expression is accepted. For example, a transform can
read time directly from state:

```python
scheduled = zdx.Map(x=state.ref("time"), s=2.0)

value = scheduled.resolve(state=state)
state = state.step()
next_value = scheduled.resolve(state=state)
print(value, next_value)
```

```text
0.2 0.4
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
