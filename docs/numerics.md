# Numerics walkthrough

The numerics package represents unrealised values as composable Equinox PyTrees.
Arrays remain optimiser leaves and evaluation occurs only when the required local
context is available.

Optional numerical operands default to `None`, so Equinox omits absent fields from
its normal representation. The printed tree shows only the active composition.

## 1. Call and resolve

Calling an expression and resolving it are equivalent. Modules recursively apply
the same protocol to their complete definition:

```python
value = expression(state=state)
value = expression.resolve(state=state)
model = model.resolve(state=state, coords=coords)
```

Calling a transform with an explicit value applies it to that value. Calling a
transform with no value resolves its stored `x`:

```python
value = zdx.Exp()(latent)
value = zdx.Exp(latent)()
value = zdx.Exp(latent).resolve()
```

All three forms return `exp(latent)`.

`Exp` and `Log` accept an optional `base`; `None` selects the natural base and is
omitted from the print. `Exp10` and `Log10` provide compact base-ten definitions:

```python
decimal = zdx.Exp(x=latent, base=10.0)
decimal = zdx.Exp10(x=latent)
```

Their unbound prints are `Exp()`, `Exp(base=f32[])`, and `Exp10()` respectively.

```python
definition = zdx.Exp(
    x=zdx.Map(
        x=latent,
        M=matrix,
        b=origin,
    )
)

print(definition)
value = definition()
```

```text
Exp(x=Map(x=f32[4], M=f32[4,64], b=f32[64]))
```

For example, an unbound `Exp` prints as `Exp()`, and `Map(M=matrix)` prints as
`Map(M=f32[4,64])` rather than listing its absent `x`, `s`, and `b` leaves.

Calling an unbound transform applies its full nested spine and returns an array:

```python
transform = zdx.Mul(x=zdx.Exp(), s=2.0)
value = transform(latent)
```

Stored composition is expressed directly through `x`; aliases expose shorter domain
names for structural paths when desired.

## 2. Runtime resolution

The owner of a field can retain its expression until authoritative runtime context
exists:

```python
class Stage(zdx.Module):
    response: object = zdx.field(runtime=True)

    def __call__(self, coords, **context):
        stage = self.resolve(runtime=True, coords=coords, **context)
        return stage.response
```

Ordinary `model.resolve(**context)` treats runtime fields as opaque. Passing
`runtime=True` explicitly overrides that protection throughout the selected tree.
This keeps a root-level coordinate array from resolving fields that belong to local
runtime coordinate systems.

The lifecycle terminology is:

1. the canonical, serialised **unrealised model**;
2. the **linked model** returned by `populate()`;
3. a **context-resolved model** whose runtime fields may remain unrealised;
4. a **realised model** containing concrete array-valued leaves.

## 3. Coordinates and parameterisations

```python
grid = zdx.Grid(n=(64, 64), d=100.0, unit="nm")
coords = grid()

polynomial = zdx.Polynomial(degree=2, ndim=2, x=coefficients)
surface = polynomial.resolve(coords=coords)

state = zdx.State(time=times[0])
time_ref = state.ref("time")
history = zdx.Interpolation(times, samples, x=time_ref)
field = zdx.Basis(M=modes, x=history)
state = state.set(time=1.5)
value = field.resolve(state=state)
```

`Grid` uses `ij` indexing: component `i` varies along spatial axis `i`, and the
coordinate output has shape `(ndim, *grid.n)` after any metadata batch axes.

`Basis` infers one unique matching coefficient shape by default. A Grid `unit`
string converts both `d` and `c` to the active global unit. Supply a `Unit` object
to choose an explicit output:

```python
grid = zdx.Grid(
    n=(64, 64),
    d=0.1,
    unit=zdx.Unit("arcsec", to="mas"),
)
```

The reusable basis, polynomial, interpolation, and profile definitions live in the
`zodiax.numerics.parametric` subpackage while remaining available at the package
root. The general profile is covariance-defined:

```python
profile = zdx.Gaussian(cov=covariance)
values = profile.resolve(coords=coords)
```

## 4. Normalisation

```python
transmission = zdx.MeanNorm(
    x=zdx.Add(
        x=zdx.Basis(M=modes, x=coefficients),
        b=1.0,
    ),
    w=support,
)

opd = zdx.RMSNorm(
    x=zdx.Basis(M=modes, x=coefficients),
    s=zdx.Unit("nm", x=10.0),
    w=support,
)
```

The default `axis=None` operates over the complete array. `MeanNorm`, `RMSNorm`, and
`SumNorm` target one when `s` is omitted.

## 5. Sharing, state, and random values

```python
distance = zdx.Linked(2.0, key="secondary-position")
reference = distance.defer()
realised = model.populate().resolve()

state = zdx.State(
    index=0,
    time=0.0,
    dt=0.1,
    key=jr.key(0),
)
scheduled = zdx.Mul(x=state.ref("time"), s=2.0)
current = scheduled.resolve(state=state)
state = state.step()
following = scheduled.resolve(state=state)

noise = zdx.Random(jr.normal, shape=(4,), stream="noise")
sample = noise.resolve(state=state)
```

The unresolved model remains the serialisable and optimisable definition. Population
and resolution produce execution artefacts. `State.step()` increments its available
index and time while retaining `key` as a stable root. `Random` derives ordinary JAX
keys from that root, the current index, and its stable stream identity, so model
components do not split keys manually.
