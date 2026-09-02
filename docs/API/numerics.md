# Numerical expressions

Zodiax stores composable numerical definitions as ordinary Equinox PyTrees and
resolves them when local context is available.

## Core protocol

- `Expression.evaluate(**context)` defines a deferred value.
- `resolve(value, **context)` recursively evaluates expressions.
- `Transform` adds stored input `x`, local `fwd(x)`, and optional reverse methods.
- Calling an expression with context is equivalent to resolving it.
- Calling a transform with an input applies its complete nested spine.
- Calling a transform without an input resolves its stored definition.
- Fields declared with `zdx.field(runtime=True)` remain protected until resolution
  explicitly supplies `runtime=True`.

Optional numerical operand fields have the dataclass default `None`. Equinox's
ordinary representation omits those absent fields, so `Exp()` and `Map(M=matrix)`
remain compact while populated leaves retain their real structural paths.

```python
transform = zdx.Mul(x=zdx.Exp(), s=2.0)
value = transform(latent)

definition = zdx.Exp(x=zdx.Map(x=latent, M=matrix, b=origin))
value = definition()

direct = zdx.Exp()(latent)
stored = zdx.Exp(latent)()  # equivalent to direct
resolved = zdx.Exp(latent).resolve()
```

Persistent composition is ordinary construction under `x`.

## Runtime fields and model lifecycle

Runtime scope belongs to the field owner rather than the expression type. The same
parameterisation may be globally resolvable in one model and require local runtime
coordinates in another:

```python
class Stage(zdx.Module):
    response: object = zdx.field(runtime=True)

    def __call__(self, coords, **context):
        realised = self.resolve(runtime=True, coords=coords, **context)
        return realised.response
```

Ordinary `model.resolve(**context)` leaves every runtime field opaque. A deliberate
`model.resolve(runtime=True, **context)` overrides protection throughout the
selected tree.

The canonical **unrealised model** becomes a **linked model** after `populate()`, a
**context-resolved model** after available high-level context is applied, and a
**realised model** once runtime fields have also become concrete arrays. Protected
subtrees are resolved coherently at their owning runtime boundary rather than being
partially evaluated beforehand.

## Operations

| Object | Kernel | Definition |
|---|---|---|
| `Add(b, x=None)` | `add(x, b=None)` | `x + b` |
| `Mul(s, x=None)` | `mul(x, s=None)` | `x * s` |
| `Pow(p, x=None)` | `power(x, p)` | `x**p` |
| `Exp(x=None, base=None)` | — | `exp(x)` or `base**x` |
| `Log(x=None, base=None)` | — | natural or selected-base logarithm |
| `Exp10(x=None)` / `Log10(x=None)` | — | compact base-ten pair |
| `MatMul(M, x=None)` | `matmul(x, M=None)` | final/leading-axis contraction |
| `Mask(mask, x=None)` | — | sparse scatter |
| `Map(x=None, s=None, M=None, b=None)` | — | `matmul(s * x, M) + b` |

`Exp` and `Log` use the natural base when `base=None`; an explicit base is real,
positive, and unequal to one. `Exp10` and `Log10` are the compact named base-ten
forms. `Add`, `Mul`, the exponential pairs, and the logarithm pairs provide exact
local inverses on their valid domains. `MatMul`, `Mask`, `Basis`, and `Map` provide
representative solves where appropriate.

## Bases and polynomials

These reusable parameterisations live in `zodiax.numerics.parametric` and remain
re-exported as `zdx.Basis`, `zdx.Polynomial`, and `zdx.Interpolation`.

`Basis(M, x=None, axes=None)` automatically matches one unique trailing coefficient
shape against leading dimensions of `M`. `axes` is an advanced override for an
ambiguous layout.

```python
field = zdx.Basis(M=modes, x=coefficients)
```

`Polynomial` stores exponent array `p` directly and evaluates its terms from
`coords`:

```python
polynomial = zdx.Polynomial(degree=2, ndim=2, x=coefficients)
value = polynomial.resolve(coords=coords)
```

`Gaussian(cov=...)` provides the general covariance-defined profile. It supports
the same one- or higher-dimensional coordinate layout produced by `Grid`:

```python
profile = zdx.Gaussian(cov=covariance)
values = profile.resolve(coords=coords)
```

## Grid

`Grid(n, d, c=None, unit=None)` is a coordinate expression with dimension-generic
`ij` indexing. Component `i` varies along sample axis `i`, so the output shape is
`(ndim, *n)` after any metadata batch axes. `unit` may be a string, which targets the
active global unit, or an unbound `Unit` with an explicit output.

```python
grid = zdx.Grid(n=(64, 64), d=0.1, unit="mm")
coords = grid()  # (2, 64, 64)

image_grid = zdx.Grid(
    n=(64, 64),
    d=0.01,
    unit=zdx.Unit("arcsec", to="mas"),
)
```

The class exposes `axes`, `coordinates`, `shape`, `batch_shape`, `fov`, `measure`,
`broadcast`, `match_shape`, `resize`, `axes_for`, and `from_axes`.

## Normalisation

`MeanNorm`, `RMSNorm`, and `SumNorm` scale their input to target `s`, which defaults
to one. Optional nonnegative `w` supplies weights or a support mask. `axis=None`
operates over the complete array.

```python
transmission = zdx.MeanNorm(
    x=zdx.Add(x=zdx.Basis(M=modes, x=coefficients), b=1.0),
    w=support,
)

opd = zdx.RMSNorm(x=field, s=zdx.Unit("nm", x=10.0), w=support)
```

## Interpolation, links, state, and random values

```python
state = zdx.State(
    index=0,
    time=0.0,
    dt=0.1,
    key=jr.key(0),
)
time_ref = state.ref("time")
history = zdx.Interpolation(times, samples, x=time_ref)
value = history.resolve(state=state)
state = state.step()
next_value = history.resolve(state=state)

owner = zdx.Linked(value, key="shared")
reference = owner.defer()
realised = model.populate().resolve(state=state)

noise = zdx.Random(jr.normal, shape=(8,), stream="noise")
sample = noise.resolve(state=state)
```

Population establishes shared ownership; resolution evaluates expressions. A
`State` owns values and uses the ordinary Zodiax `set` path API. `StateRef` stores
only a static path and may occupy any expression leaf, so interpolation does not
need a separate named-argument selector. `State.step()` advances `index` and `time`
without mutating the root `key`. `Random` wraps any JAX-style key-first callable and
derives an ordinary JAX key from the root key, current index, and its stable stream
identity, so consuming model code does not split keys manually.

Common JAX trace-time options, including `shape`, `dtype`, and `axis`, are static
automatically. Use `static=` to name any additional trace-time keyword required by
a custom key-first callable.

## Units and arrays

```python
zdx.set_units({"cartesian": "um", "angular": "mas"})
distance = zdx.Unit("mm", x=2.0)
distance()  # 2000 um

explicit = zdx.Unit("mm", to="m", x=2.0)
explicit()  # 0.002 m
```

Resolved values remain ordinary JAX arrays. `as_array` converts numerical inputs
while preserving `None` and `Expression` objects at model-constructor boundaries.
