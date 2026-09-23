# Module and aliases

Module adds model preparation and concise parameter lookup to Base's immutable path operations.

`Module`, `Alias`, `update`, and `validate_aliases` are implemented in
`zodiax.module` and exported directly from `zodiax`.

## 1. Reaching a model's parameters

A Module exposes uniquely named descendant fields through the nearest matching child.

```python
import jax
import jax.numpy as np
import zodiax as zdx


class Surface(zdx.Module):
    coefficients: jax.Array | zdx.Expression


class Optic(zdx.Module):
    surface: Surface


optic = Optic(surface=Surface(coefficients=np.array([0.1, 0.2])))
print(optic)
print(optic.coefficients)

updated = optic.set(coefficients=np.array([0.3, 0.4]))
print(updated.coefficients)
print(optic.coefficients)
```

```text
Optic(surface=Surface(coefficients=f32[2]))
[0.1 0.2]
[0.3 0.4]
[0.1 0.2]
```

This is raised lookup. A declared attribute on the current object takes precedence.
Otherwise a local alias is used, followed by the nearest matching descendant.
Several matches at the same depth are ambiguous; use a qualified path to select one.
Raised lookup also includes aliases declared on descendant Modules. Path updates
return new objects and leave the original unchanged.

## 2. Giving structural paths local names

An alias assigns a name to an existing dynamic structural path without adding another parameter leaf.

```python
named = Optic(
    surface=optic.surface,
    alias={"amplitudes": "surface.coefficients"},
)
print(named)
print(named.alias)

adjusted = named.set(amplitudes=np.array([0.5, 0.6]))
print(adjusted.amplitudes)
```

```text
Optic(
  alias=(('amplitudes', 'surface.coefficients'),),
  surface=Surface(coefficients=f32[2])
)
(('amplitudes', 'surface.coefficients'),)
[0.5 0.6]
```

Alias input may be a mapping, one `(name, path)` pair, a sequence of pairs, or
`None`. The stored `Alias` type is `tuple[tuple[str, str], ...]`; construction sorts
entries by name and stores an empty specification as `None`. This metadata is static
in the JAX tree, so equivalent mappings have the same canonical representation.

Names must be unique public Python identifiers and cannot collide with declared
attributes. A local alias may intentionally select one otherwise ambiguous
descendant. Its target follows dynamic fields on Equinox Modules, mapping keys, or
canonical non-negative list/tuple indices, such as `surfaces.0.coefficients`.
Targets cannot cross static fields, properties, another alias, or raised shorthand.
Different alias names may target the same field.

## 3. Preparing definitions

Resolution preserves an ordinary Module while replacing ready Expression fields with their values.

```python
definition = Optic(
    surface=Surface(coefficients=zdx.Map(x=[0.1, 0.2], s=2.0)),
    alias={"latent": "surface.coefficients.x"},
)
prepared = definition.resolve()
assert isinstance(prepared, Optic)
assert isinstance(prepared.surface.coefficients, jax.Array)
assert np.allclose(prepared.surface.coefficients, np.array([0.2, 0.4]))
assert np.allclose(definition.latent, np.array([0.1, 0.2]))
```

The alias `latent` describes the original Map's stored input. The prepared array has
no `x` field, so `prepared.latent` raises `AttributeError`, and explicitly validating
that prepared alias with `prepared.validate_aliases()` raises `ValueError`.
Resolution does not rewrite alias targets. Edit the definition using its aliases,
then resolve it for calculation.

`resolve` connects shared links by default and retains expressions whose required
context is unavailable. `populate` performs only structural link replacement.
See the [numerics walkthrough](../numerics.md) for partial evaluation and local context.

## 4. Updating several models

`update` distributes named path updates across an ordered collection of Modules.

```python
second = named.set(amplitudes=np.array([0.7, 0.8]))
parameters = {"amplitudes": np.array([1.0, 2.0])}

first_only, untouched = zdx.update(parameters, named, second)
assert np.allclose(first_only.amplitudes, parameters["amplitudes"])
assert np.allclose(untouched.amplitudes, second.amplitudes)

first_all, second_all = zdx.update(parameters, named, second, mode="all")
assert np.allclose(first_all.amplitudes, parameters["amplitudes"])
assert np.allclose(second_all.amplitudes, parameters["amplitudes"])
```

Default `mode="first"` assigns each path to its first matching object. `mode="all"`
updates every match. The returned tuple preserves object order. `strict=True`
rejects paths unused by every object; `strict=False` permits unused paths.

## 5. Developer contracts

The following contracts distinguish static aliases, convenient reads, and structural updates.

| API and input | Output | Contract |
|---|---|---|
| Constructor `alias=None` or an empty specification | Stored `None` | No local aliases. |
| Constructor alias mapping, pair, or sequence of string pairs | Static `Alias` tuple | Sort by name; require unique public identifiers and available dynamic targets. |
| Declared attribute read | Its value | Declared attributes precede aliases and descendant lookup. |
| Local alias read | Value at its structural target | No duplicate parameter leaf; stale targets raise `AttributeError`. |
| Raised descendant read | Value at the nearest matching depth | Equal-depth ambiguity raises `AttributeError`; use qualified paths. |
| `get(path, to_array=False)` | Value of any supported type | Paths may select local aliases and uniquely raised names. |
| Inherited `set(path, value)` or keyword updates | New Module | No value conversion or constructor revalidation; callers supply valid replacements. |
| `validate_aliases(tree)` or `module.validate_aliases()` | `None` or exception | Check current alias topology explicitly, including nested Modules. |
| Module construction containing mappings | Valid Module or exception | Dots are reserved for path separators and cannot occur in string mapping keys. |
| `resolve(**context)` on an ordinary Module | Same Module class with prepared fields | Populate links by default; preserve unavailable local definitions. |
| `update(mapping[str, Any], *modules, mode="first")` | Tuple of Modules | Each path updates its first match; `mode="all"` updates every match. |
| `update(..., strict=True)` with unused paths | `KeyError` | `strict=False` allows paths that match no object. |

Choose aliases outside JAX-transformed loops because they are static tree metadata.
Whole-subtree updates or resolution can invalidate targets; explicit validation
checks the resulting topology rather than repairing it.

::: zodiax.module
