# Numerics design

## Principles

1. The unresolved Equinox PyTree is the single definition used by optimisation,
   aliases, layouts, and serialisation.
2. Calling an expression resolves it; a transform call may instead supply an
   explicit deepest input.
3. Persistent composition is ordinary construction through `x`.
4. A small set of mathematical nodes is preferred over composition containers,
   callable wrappers, or semantic duplicate fields.
5. Aliases provide downstream vocabulary without requiring remapping subclasses.
6. Coordinate arrays are the universal consumer contract; `Grid` is an optional
   producer of those arrays.
7. Optional numerical leaves default to `None`, allowing Equinox to omit inactive
   operands from the ordinary printed representation without a global repr override.
8. Runtime scope belongs to the field owner. Ordinary resolution preserves declared
   runtime fields until an explicit `runtime=True` evaluation boundary.
9. Model definitions remain stable while external State carries time, iteration,
   and the root of deterministic JAX random streams.

## Object model

```text
Module
├── State
└── Expression
    ├── Linked, Deferred
    ├── StateRef
    ├── Random
    ├── Grid
    └── Transform
        ├── Add, Mul, Pow, Exp, Log, Exp10, Log10
        ├── MatMul, Mask, Map
        ├── parametric/
        │   ├── Basis, Polynomial, Interpolation
        │   └── Gaussian
        ├── Norm, MeanNorm, RMSNorm, SumNorm
        └── Unit
```

`evaluate(**context)` defines one expression's evaluation and `fwd(x)` describes one
transform node. `expression(**context)` and `resolve(expression, **context)` are
equivalent. `transform(x)` applies the complete nested spine to an explicit deepest
input, while `transform()` resolves its stored input.

Fields declared with `zdx.field(runtime=True)` are opaque to ordinary recursive
resolution. `resolve(runtime=True, **context)` is the explicit override used by the
owning runtime boundary. Protection belongs to the downstream field rather than the
expression class, so the same expression can be runtime-dependent in one model and
globally resolvable in another.

The canonical unrealised model is serialised and optimised. Population produces a
linked model, ordinary context evaluation produces a context-resolved model, and
runtime evaluation produces a realised model whose numerical definitions have been
replaced by concrete arrays.

`TreeLayout`, `TreeVector`, `TreeMatrix`, `Jacobian`, `Hessian`, and `Fisher` live in
the sibling `zodiax.linalg` module. They describe realised PyTree coordinate spaces
and derivative arrays rather than deferred numerical expressions.

## Basis and polynomial rules

`Basis` matches one unique trailing shape of coefficients against leading basis
dimensions. Ambiguity raises and can be resolved with the advanced `axes` override.
Matrix-free bases override the basis operation rather than materialising a dense
tensor.

`Polynomial` stores exponent array `p` and generates its own terms from contextual
`coords`. Term generation is an implementation detail, not a separate public
expression object.

## Grid boundary

`Grid` stores static `n`, numerical `d` and `c`, and an optional shared unit
conversion. A unit string targets its category's captured global output; an unbound
`Unit` can select a specific output through `to`. The Grid applies the conversion to
both `d` and `c` before generating coordinates. Grid axes use dimension-generic
`ij` indexing, so component `i` varies along sample axis `i` and the spatial shape
is exactly `n`.

dLux retains physical-category enforcement, SI conveniences, FFT conventions,
plotting extents, builders, and propagation resizing.

## Normalisation

Normalisation nodes store `x`, optional target `s`, optional weights `w`, and an
advanced static `axis`. `axis=None` reduces the complete array. Invalid zero measures
raise rather than returning infinities.

`MeanNorm`, `RMSNorm`, and `SumNorm` scale their complete input expression. Additive
centring is not represented by a separate specialised node.

## Sharing, state, and context

`State` is a sibling `Module` that owns named running values. `StateRef` stores only
a static path into the call-local State and can occupy any expression leaf; for
example an `Interpolation` can store a state-backed time query directly in `x`.
Updates use the normal `State.set(...)` path API rather than overriding it on a
reference. State topology and leaf array signatures remain fixed for JAX loop carry.
`State.step()` increments `index` and advances `time` by stored or explicit `dt`; a
`key` entry remains a stable root.

`Random` stores one JAX-style key-first callable, its arguments, and a static stream
identity. It derives an ordinary JAX key by folding the State index and stream into
the stable root. This makes random resolution independent of traversal order without
introducing a custom PRNG type or one Zodiax class per distribution. JAX trace-time
call options are retained as static metadata, while numerical call arguments remain
array leaves. `Linked` owns a model value once and `Deferred` provides static-key
uses populated from the containing model tree.

## Paths and archives

Numerical operands use `x`, `b`, `s`, `M`, and `p`. Domain names are aliases onto
those real paths. `TreeLayout` and archive definitions therefore inspect the same
topology that evaluation executes; no separate parameter graph or filled semantic
copy exists.

The `None` defaults affect presentation, not path identity. Once populated, every
operand is still an ordinary dataclass field and therefore participates normally in
aliases, layouts, optimisation, and archive definitions.
