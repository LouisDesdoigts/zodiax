# Serialisation

Save a model's declared fields and array values to a `.zdx` file.

```python
import zodiax as zdx

model = zdx.Map(x=[1.0, 2.0], s=2.0, b=1.0)
zdx.save("model.zdx", model)
restored = zdx.load("model.zdx")
print(restored)
print(restored.resolve())
```

```text
Map(x=f32[2], s=f32[], b=f32[])
[3. 5.]
```

A `pathlib.Path` works too. Paths without a suffix receive `.zdx`. The Base
conveniences are `model.save("model.zdx")` and `model.load("model.zdx")`; the latter
uses the existing model as a structural template. Saving leaves definitions
unevaluated.

## Model fields and behaviour

Model classes use the Zodiax/Equinox declared-field format: persistent state belongs
in fields containing supported arrays, literals, or containers. Loading restores
those fields on an available class without running its constructor, converters, or
user validation callbacks.

Methods come from the installed class. Python functions, lambdas, closures, and
Python code are not archived. A callable Module is supported because its state is
stored structurally and the installed class provides its `__call__` method:

```python
import jax


class Linear(zdx.Module):
    slope: jax.Array
    offset: jax.Array

    def __call__(self, x):
        return self.slope * x + self.offset


linear = Linear(slope=zdx.as_array(2.0), offset=zdx.as_array(1.0))
zdx.save("linear.zdx", linear)
restored_linear = zdx.load("linear.zdx", like=linear)
print(restored_linear)
print(restored_linear(zdx.as_array([1.0, 2.0])))
```

```text
Linear(slope=f32[], offset=f32[])
[3. 5.]
```

The archive preserves supported fields and values, not the meaning of changed class
code or process settings. For example, a Unit with `unit_out=None` uses the loading
process's current unit convention. Explicit `to=` fixes that target in the definition.
Repeated Python object identity is not preserved; use Linked/Deferred for shared
numerical ownership.

## Loading and inspecting a definition

`load(path, like=model)` checks class and field structure, container topology,
literal types, and array shapes/dtypes; stored values take precedence. Local classes
can be supplied through `like` or `custom_types={"module:qualname": Class}`. These
options are mutually exclusive, and the class identifier and declared fields must
match. Loading uses already imported or explicitly supplied classes; it does not
import code named by an archive. Classes and archives must come from trusted providers.

`strict=True` also requires matching recorded package versions. `strict=False`
skips that comparison, while schema and payload checks remain. Neither option
migrates class layouts or guarantees unchanged numerical behaviour.

`ObjectDefinition.from_object(model)` describes the structure without storing array
contents. Its `to_dict()` and `to_json()` methods expose that definition;
`build_template()` creates array placeholders, and `validate(model)` compares
metadata rather than numerical values.

## Developer contract

A version-1 archive contains a JSON manifest and a JAX array payload. Saving and
loading are host operations outside JIT; device placement follows the loading
process. The following tests live in `tests/serialisation/`.

| Input or operation | Output and guarantee | Tests |
|---|---|---|
| `save(path, tree)` / `load(path)` | File / reconstructed supported tree; path writes replace atomically. Binary streams are also accepted and remain open; final stream-copy failures can leave partial output. | `test_archive.py` |
| `zdx.Module`, `zdx.Base`, or other `eqx.Module` | Installed class with its supported declared fields restored; no undeclared instance state or constructor replay. Callable Modules follow this same rule. | `test_definition.py`, `test_extensions.py` |
| Exact `eqx.nn.State` | Restored supported values and stable integer/string state keys; use `eqx.nn.make_with_state` to construct stateful models. State subclasses are unsupported. | `test_extensions.py` |
| list, tuple, dict, OrderedDict | Corresponding supported container and contents; ordinary dicts use JAX's canonical order, OrderedDict retains insertion order. | `test_definition.py` |
| Literals and static fields | None, bool, signed 64-bit Python int, finite float/complex, str, range, slice, and available type references retain their represented values. Static fields cannot contain arrays or Linked/Deferred nodes. | `test_definition.py`, `test_extensions.py` |
| Strong concrete JAX arrays and typed PRNG keys | Supported shape, dtype, values, and key representation preserved. Incompatible precision settings raise instead of narrowing. NumPy leaves and weak JAX arrays are unsupported. | `test_definition.py`, `test_archive.py` |
| Standalone functions, closures, or opaque objects in fields | Rejected; store supported state and implement behaviour as methods on the model class. | `test_definition.py`, `test_edge_cases.py`, `test_extensions.py` |
| Module paths, aliases, and dynamic link ownership | Structural checks before saving and after loading; Base metadata retains its ordinary rules. No numerical evaluation is implied. | `test_extensions.py`, `test_edge_cases.py` |
| ObjectDefinition, `like`, and `custom_types` | Validated metadata and explicit class resolution; malformed schemas and inconsistent or incomplete payloads raise. | `test_definition.py`, `test_archive.py` |

Array storage supports bool; signed/unsigned 8, 16, 32 and 64-bit integers;
float16, bfloat16, float32, float64; complex64 and complex128. Archive fidelity
covers this stored representation, independently of numerical default precision.
