# Serialisation

Zodiax archives store an automatically generated `ObjectDefinition` beside an
Equinox array payload. The definition records module types, declared fields,
containers, static metadata, and array shape/dtype information. It is inspectable
JSON and reconstructs modules without calling their constructors.

```python
import jax.numpy as jnp
import zodiax as zdx

owner = zdx.Linked(jnp.array([1.0, 2.0]), key="shared-coefficients")
model = zdx.Map(
    x=owner.defer(),
    s=1e3,
    M=jnp.eye(2),
)

zdx.save("model", {"owner": owner, "value": model})
restored = zdx.load("model")
```

Every `Base` also exposes the same workflow as `model.save("model")` and
`model.load("model")`; the latter uses `model` as a structural template.

Paths without a suffix receive `.zdx`. Loading validates the ZIP container, manifest,
object definition, and complete payload consumption. It never imports code named by
an archive: stored classes must already be imported, supplied by a trusted `like`
object, or explicitly registered through `custom_types`.

By default, `load(..., strict=True)` also requires exact matches for every recorded
package version. `strict=False` permits an explicit best-effort reconstruction when
the archive format and generated object definition remain compatible. Python's
version is retained as diagnostic provenance but is not part of this package-version
check.

Plain `dict` objects follow JAX PyTree semantics and are reconstructed in canonical
key order. Use `collections.OrderedDict` when insertion order is part of the stored
object's meaning.

Equinox may omit a numerical field whose value matches its `None` default from the
printed representation. This is presentation only: the declared field and its
`None` value remain part of the generated object definition and round-trip normally.
Behaviour-affecting Zodiax field metadata, including runtime-resolution protection,
is stored in the definition and checked against the installed class during loading.

Saving and loading also validate complete link topology. Duplicate `Linked` owners,
dangling `Deferred` keys, and dependency cycles are rejected; archive the containing
unrealised model rather than a detached deferred fragment.
`Linked` and `Deferred` nodes are rejected beneath static Equinox fields: static
subtrees are excluded from JAX traversal and therefore cannot participate in link
population.

## Callables and state

Archives never derive an import from a stored callable name. Stateless functions can
opt into portable serialisation through an explicit symbolic registry:

```python
@zdx.register_callable("my_package.activations.square")
def square(value):
    return value**2
```

The same registration must run before loading in a new process. Unregistered lambdas
and functions are rejected. Callable PyTrees with numerical leaves should normally be
named Equinox or Zodiax modules; the registry represents the stateless function, not
captured mutable state. The default callables used by `equinox.nn.MLP` are registered
by Zodiax.

Zodiax `State` is an ordinary serialisable Module, so a model, running state, and
next JAX key can be checkpointed together:

```python
zdx.save("checkpoint", {"model": model, "state": state, "key": next_key})
checkpoint = zdx.load("checkpoint")
zdx.validate_state(checkpoint["model"], checkpoint["state"])
```

`StateRef` paths are static strings and `validate_state` checks them against the
loaded State. JAX keys remain ordinary explicit model inputs or array leaves, and
typed JAX keys round-trip. Persist the newest State and the next unused application
key between completed steps.

## Reconstruction and validation

Equinox modules are rebuilt without invoking their constructors. Classes with
value-dependent invariants can define a no-argument `__zodiax_validate__` hook; it is
called after the complete object is loaded, must be pure, and should raise when stored
state is invalid. Every class-local hook is invoked from base to derived, so a hook
validates only the fields introduced by its declaring class and should not call
`super()`. Zodiax validates the ZIP members, bounded manifest, object schema, array
headers, and complete payload consumption before returning an object. Loading a
genuinely large payload still requires memory proportional to its stored array data,
so applications accepting external archives may impose their own file-size limit.

Exact package versions do not make actively changing local classes durable. Local
classes may be resolved with `like` or `custom_types`, and installed distributions
referenced by stored classes are recorded when discoverable, but stable long-lived
archives require stable package releases and class field schemas. Avoid treating
checkpoints of actively developed local objects as permanent artifacts.

`TreeLayout` and `ObjectDefinition` provide complementary views of a model tree.
`ObjectDefinition` is the complete serialisation schema; `TreeLayout` independently
describes the ordered path-and-shape projection of an explicit parameter tree.
Layouts and their derivative vectors or matrices are ordinary Equinox modules, so
the generated object definition serialises them directly—there is no parallel
`value_spec` or `latest_spec` state to synchronise. Their traversal implementations
remain separate because one describes every serialisable field and the other
describes the parameters supplied to a derivative calculation.

::: zodiax.serialisation
