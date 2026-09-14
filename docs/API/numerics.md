# Numerical expressions

Zodiax stores composable numerical definitions as ordinary Equinox PyTrees. Use
`Map` for scale, matrix or basis contraction, and bias:

```python
import jax
import jax.numpy as np
import jax.random as jr
import zodiax as zdx

latent = np.array([-1.0, 0.0, 1.0])
definition = zdx.Map(x=latent, s=2.0, b=1.0)
value = definition.resolve()
same = definition()

print(definition)
print(value)

operation = zdx.Map(s=2.0, b=1.0)
value = operation(latent)
```

```text
Map(x=f32[3], s=f32[], b=f32[])
[-1.  1.  3.]
```

The printed array dtypes use JAX's default configuration. The
[numerics design](../numerics_design.md) shows the complete class hierarchy and
package layout; the [walkthrough](../numerics.md) develops runnable examples.

## Core protocol

- `model.resolve(**context)` connects shared links and prepares available definitions.
  It is the normal evaluation entry point; link population is enabled by default.
- An ordinary Module retains its class while its fields prepare. An Expression
  becomes its value when ready or remains partially prepared when inputs are missing.
- `Expression.evaluate(**context)` implements one expression's calculation.
- `Transform` adds stored input `x`, local `fwd(x)`, and optional local `inv(y)`.
  Its inherited `evaluate` supplies the stored input to `fwd`.
- `transform(x)` and `transform.apply(x)` apply the full nested Transform chain to
  an explicit deepest input. `transform()` resolves the stored definition.
- `transform.apply(y, inverse=True)` reverses that chain; the explicit call
  `transform(y, inverse=True)` is equivalent.
- `Operation` is a Transform for elementwise formulas using ordinary JAX broadcasting.

Optional numerical operands default to `None` and are omitted by Equinox's normal
representation. Nested transforms express calculations with different operations:

```python
operation = zdx.Exp(x=zdx.Map(s=2.0, b=1.0))
value = operation(latent)
recovered = operation.apply(value, inverse=True)
print(operation)
print(recovered)
```

```text
Exp(x=Map(s=f32[], b=f32[]))
[-1.  0.  1.]
```

`fwd` and `inv` operate locally on prepared arrays. Forward `apply` works from the
deepest transform outwards; reverse `apply` follows the chain in the opposite order.
Keep `inverse` a Python boolean fixed during JAX tracing. The advanced
`initialise(y)` helper binds the recovered deepest input in a new definition.

## Transform classes

| Class | Definition | Reverse |
|---|---|---|
| `Map(x=None, s=None, M=None, b=None)` | `matmul(s * x, M) + b`; omitted operands skip their stage | Representative `inv` |
| `Mask(x=None, mask=None)` | Scatter compact values into selected entries | Gather with `inv` |
| `Exp(x=None, base=None)` | `exp(x)` or `base**x` | Principal logarithm with `inv` |
| `Log(x=None, base=None)` | Natural or selected-base logarithm | `inv` on the selected branch |
| `Pow(x=None, p=None)` | `x**p` | Forward-only |
| `Unit(x=None, unit=None, to=None)` | Convert to an explicit or global output unit | `inv` |

Exp, Log, Pow, and Unit inherit Operation. Their array operands follow ordinary JAX
broadcasting, including shape expansion. Elementwise inverse results retain that
broadcasted shape rather than reducing to a smaller original input shape. Explicit
logarithmic bases must be real, positive, and unequal to one; complex logarithms use
the principal branch.

Map scale and bias may broadcast only without expanding the relevant shape.
`matmul(x, M)` contracts `(*batch, n)` with `(n, *output)` to produce
`(*batch, *output)`; `M=None` is identity. Use `Map(M=matrix)` for a matrix-only
transform. Map reverses the stages by subtracting bias, applying a numerical
pseudoinverse with JAX's native rank cutoff, and dividing by scale. Nonzero scales
are required for division. This stagewise representative need not recover an
arbitrary input or output exactly, nor minimise the original input norm when a
nonuniform scale precedes a rank-deficient matrix.

Construct Mask before entering JIT so its selected count is known. It stores a
dynamic boolean array and replaces the final compact axis with `mask.shape`,
retaining batch axes. Updating the mask must
preserve its count. When passed dynamically through JIT, its selected indices are
recomputed with work proportional to the number of mask elements.

## Partial resolution

Missing context retains an expression with its available children prepared:

```python
definition = zdx.Map(x=zdx.StateRef("input"), s=zdx.Map(x=2.0, b=1.0))
prepared = definition.resolve()  # Scale is 3; input still requires state.
state = zdx.State(input=[0.0, 1.0, 2.0])
print(prepared)
print(prepared.resolve(state=state))
```

```text
Map(x=StateRef(path='input'), s=f32[])
[0. 3. 6.]
```

Required context is declared in `evaluate` for Expressions or `fwd` for Transforms.
The calculation resolves the operands it actually uses. If it cannot finish,
inherited child preparation keeps available progress. Other errors propagate.

Each scientific context name must retain one meaning throughout the tree. Use
distinct names for incoming and derived coordinates. The fallback forwards supplied
context unchanged; it cannot infer coordinate frames. Reusing the original definition
reads changing context, while a prepared copy retains values already resolved.

## Shared values and state

```python
class Measurements(zdx.Module):
    science: jax.Array | zdx.Expression
    reference: jax.Array | zdx.Expression


gain = zdx.Linked(2.0, key="gain")
model = Measurements(
    science=zdx.Map(x=3.0, s=gain),
    reference=zdx.Map(x=1.0, s=gain.defer()),
)
prepared_model = model.resolve()
science = prepared_model.science      # 6.0
reference = prepared_model.reference  # 2.0
```

Place one Linked owner in the tree and use Deferred references elsewhere. Owners
may contain arrays, Expressions, or PyTrees. Resolution collects all owners before
replacing links. Missing or duplicate owners and cycles are errors. Keep resolution
inside differentiated calculations so gradients accumulate at the original owner.

`populate` is an advanced structural-only operation. `validate_links` checks the same
topology without evaluating Expressions. `resolve(populate=False)` skips automatic
population when the caller has already prepared the links.
Automatic population covers the complete tree passed to `resolve`. Before calling
an explicit-input operator whose operands reference one another, resolve the unbound
operator or its containing Module, then call the prepared operator.

```python
state = zdx.State(index=0, time=0.0, dt=0.1, key=jr.key(0))
scheduled = zdx.Map(x=state.ref("time"), s=2.0)
value = scheduled.resolve(state=state)
state = state.step()
next_value = scheduled.resolve(state=state)
```

State converts each numerical entry to one array; expressions belong in the model.
StateRef stores a path and reads from the supplied State. Inherited `set` does not
convert replacements, so supply arrays and preserve loop-carry shapes, dtypes, and
tree structure. `step` reads the original index and time before updating them;
random keys and other entries remain unchanged.

## Units and arrays

```python
distance = zdx.Unit(x=2.0, unit="mm")
print(distance)
print(distance.resolve())
print(distance.to("um"))
```

```text
Unit(x=f32[], unit='mm')
0.002
2000.0
```

Unit stores an input `unit` and optional `unit_out`. The default `unit_out=None`
follows global settings when evaluated; an explicit `to=` fixes the target.
`to(unit, **context)` returns a converted array without changing the definition;
`to(None)` converts into the current global convention. It raises `ValueError` if
the stored input is absent or cannot finish with the supplied context. Eager calls
read current settings; JIT retains the settings captured when tracing, so cached
compiled calls do not change after `set_units`. Set the convention before tracing
or retrace after changing it.

`set_units` resets omitted categories to built-in defaults. `get_units` and
`set_units` return independent dictionaries. Explicit-unit conversion helpers remain
independent of global settings.

`as_array` converts numerical inputs to strongly typed JAX arrays. Inferred dtypes
are preserved unless `float`, `int`, `complex`, or `bool` is requested; JAX's
configuration chooses precision. None and Expressions pass through unchanged, with
no future cast attached. Realised numerical values remain ordinary arrays.

## Reference

::: zodiax.numerics.expressions

::: zodiax.numerics.transforms

::: zodiax.numerics.operations

::: zodiax.numerics.links

::: zodiax.numerics.units

Array and State references are on the [arrays and state](stateful.md) page.
