# Base

`Base` provides Zodiax's immutable path and archive operations. `Module` is an opt-in
child class for composite numerical models: it adds `populate`/`resolve`, local
aliases, and raised descendant fields. Equal-depth ambiguities are rejected with
qualified paths instead of silently depending on child order.

`zdx.field(runtime=True)` marks a numerical field whose owner must supply its local
runtime context. Ordinary `resolve()` calls retain that subtree; an explicit
`resolve(runtime=True, **context)` evaluates it. Runtime metadata is part of the
field declaration, is absent from ordinary object prints, and is restored naturally
with the owning class during archive loading.

`Module(alias=...)` adds local, static names for canonical structural paths. Alias
names take precedence over raised descendants, which makes them useful for resolving
otherwise ambiguous shorthand, but they may not shadow a field declared by the
owning object. Targets must traverse dynamic dataclass fields, mapping keys, or
canonical non-negative sequence indices; aliases cannot target another alias,
property, raised name, or static field.

Raised descendant attributes remain a convenient read API and can include public
properties or static metadata. Aliases are the stricter update API: because their
targets are dynamic structural paths, they are safe selectors for `get`, `set`,
optimisation layouts, and serialisation.

```python
import zodiax as zdx

class Model(zdx.Module):
    output: object


model = Model(
    output=zdx.Exp(x=zdx.Mul(x=zdx.Add(x=latent, b=offset), s=scale)),
    alias=(("offset", "output.x.x.b"),),
)

model.offset          # concise lookup
model.get("offset")  # the same path through the Zodiax API
```

`alias=None` is the inherited default and is omitted by Equinox's ordinary printed
representation. A non-empty alias is static PyTree metadata and is printed normally;
changing it therefore changes the JAX tree definition. Call `validate_aliases()`
after topology-changing updates. Construction, saving, and loading perform their own
central validation; populated and resolved execution copies may intentionally have a
different topology and should not be archived.

`Base.save(path)` writes a validated Zodiax archive. Calling `template.load(path)`
loads through the same format while using `template` to verify the expected class and
field structure.

Dots are reserved as path separators. `Base` validates stored mappings after
construction and `set()`, and rejects literal string keys containing dots. Dotted
strings remain valid as mapping values and static metadata, including alias targets
and `StateRef` paths. Archive reconstruction repeats the same validation before an
object is returned.

::: zodiax.base
