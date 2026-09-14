"""Public save and load workflows for validated Zodiax archives."""

from collections.abc import Mapping
from contextlib import nullcontext
from os import PathLike
from typing import Any, BinaryIO
import zipfile

import equinox as eqx

from ..numerics.links import validate_links
from ._archive import (
    _PAYLOAD_MEMBER,
    _PayloadReader,
    _normalise_path,
    _read_manifest,
    _save_file,
    _save_path,
    _validate_members,
    _validate_package_versions,
    _validate_payload_size,
)
from ._leaves import _deserialise_leaf
from ._validation import _validate_modules
from .definition import ObjectDefinition

__all__ = ["ObjectDefinition", "save", "load"]


def save(file_or_path: str | PathLike[str] | BinaryIO, pytree: Any) -> None:
    """Save declared model fields and supported data to a Zodiax archive.

    Parameters
    ----------
    file_or_path : str, pathlib.Path, or binary file
        Destination for the archive. Paths without a suffix receive ``.zdx``.
        Binary files remain open after saving.
    pytree : Equinox module, Equinox State, or supported container
        Model state declared in Zodiax/Equinox fields, supported built-in containers,
        JAX arrays, Python numerical values, and literals. Static fields may contain
        supported non-array values. The saved model remains unevaluated.

    Raises
    ------
    TypeError
        If any leaf or static value cannot be represented losslessly.

    Examples
    --------
    Save a model using the functional API:

    ```python
    import jax.numpy as jnp

    import zodiax as zdx

    model = zdx.Map(x=jnp.ones(2), s=2.)
    zdx.save("model", model)
    ```

    Notes
    -----
    Concrete JAX arrays are written to the binary payload. Python numerical values,
    literals, static values, and JAX array metadata are written to the generated JSON
    definition.

    NumPy array and scalar leaves are intentionally outside this initial contract.
    Standalone functions, closures, and function-valued fields are unsupported.
    Put behaviour in class methods and persistent state in declared fields. Callable
    Modules follow the same field contract as other Modules; their methods come from
    the installed class, not the archive. Opaque objects, weakly typed JAX arrays,
    and unsupported JAX dtypes cannot be saved losslessly.
    Python integers must fit in signed 64 bits. Non-finite
    Python float and complex values are unsupported and should be stored as strongly
    typed JAX arrays when required. Path destinations are replaced atomically;
    streams stage encoding first, but a later stream-copy failure can leave partial
    output. Process unit conventions and arbitrary Python object identity are not
    captured; use explicit Unit targets and Linked/Deferred ownership when needed.
    """
    # Describe first: unsupported fields and containers fail before any PyTree
    # traversal or file writes. The saved model remains unevaluated.
    definition = ObjectDefinition.from_object(pytree)
    _validate_modules(pytree)
    # Link traversal assumes a supported JAX PyTree. Running it after definition
    # construction preserves the serializer's stable diagnostics for unsupported
    # containers while still rejecting invalid links before any bytes are written.
    validate_links(pytree)

    # Stage file objects or atomically replace path destinations.
    path = _normalise_path(file_or_path)
    if path is None:
        _save_file(file_or_path, pytree, definition.to_dict())
    else:
        _save_path(path, pytree, definition.to_dict())


def load(
    file_or_path: str | PathLike[str] | BinaryIO,
    *,
    like: Any = None,
    custom_types: Mapping[str, type] | None = None,
    strict: bool = True,
) -> Any:
    """Load a validated Zodiax archive, optionally using an existing template.

    Parameters
    ----------
    file_or_path : str, pathlib.Path, or binary file
        Zodiax archive to read. Paths without a suffix receive ``.zdx``. Binary files
        must be readable and seekable and remain open after loading.
    like : Equinox module or supported container, optional
        Trusted object used to resolve classes and validate module fields, container
        topology, mapping keys, literal types, and JAX array shapes and dtypes. Its
        non-array values are ignored; archived values come from the stored definition.
        If omitted, classes are resolved from modules that are already imported.
    custom_types : mapping[str, type], optional
        Explicit mapping from stored ``"module:qualname"`` identifiers to classes.
        This can resolve local or otherwise unavailable classes without importing
        code named by the archive. The supplied class must retain the stored nominal
        identifier and field schema.
    strict : bool, default=True
        Require every package version recorded by the archive to match the loading
        environment exactly. Set to ``False`` to attempt reconstruction using the
        archive format and generated object definition alone.

    Returns
    -------
    pytree : Equinox module, Equinox State, or supported container
        Reconstructed object with its stored values restored from the archive.

    Raises
    ------
    ValueError
        If the archive is invalid, a required class is unavailable, or its structural
        definition is incompatible with ``like``.
    TypeError
        If ``like`` or ``custom_types`` contains an unsupported value.

    Examples
    --------
    Reconstruct a model directly, or restore it into a matching template:

    ```python
    import jax.numpy as jnp

    import zodiax as zdx

    restored = zdx.load("model")

    like = zdx.Map(x=jnp.zeros(2), s=1.)
    checked = zdx.load("model", like=like)
    ```

    Notes
    -----
    Archives contain an automatically generated JSON object definition and an
    Equinox JAX-array stream. The producing Python version is diagnostic provenance.
    Core package versions, plus discoverable installed distributions referenced by
    stored classes, are exact compatibility requirements when ``strict=True``.
    Archive and payload format versions and the generated definition are always
    validated.

    Loading never imports code named by an archive. For template-free loading, every
    stored class must already be imported or explicitly supplied through
    ``custom_types``. An archive can select an already imported Equinox class, so
    template-free loading should be limited to trusted archives and class providers.
    Constructors are not run when rebuilding Equinox modules, so validation checks
    stored structure and field representation rather than rerunning package-specific
    constructor invariants. Module paths and aliases and complete link ownership
    are checked centrally. No user validation callbacks run during saving or loading.
    """
    if type(strict) is not bool:
        raise TypeError("strict must be a bool.")

    # Open a path while retaining ownership of caller-provided file objects.
    path = _normalise_path(file_or_path)
    source = nullcontext(file_or_path) if path is None else path.open("rb")

    with source as file:
        try:
            with zipfile.ZipFile(file, mode="r") as archive:
                # Validate the archive and template before decoding JAX arrays.
                _validate_members(archive)
                manifest = _read_manifest(archive)
                if strict:
                    _validate_package_versions(manifest)
                _validate_payload_size(archive, manifest)
                definition = ObjectDefinition.from_dict(manifest["definition"])
                template = definition.build_template(
                    like=like,
                    custom_types=custom_types,
                )

                # Restore every JAX array leaf and require complete payload use.
                with archive.open(_PAYLOAD_MEMBER) as payload:
                    reader = _PayloadReader(payload, manifest["payload"]["size"])
                    try:
                        loaded = eqx.tree_deserialise_leaves(
                            reader,
                            template,
                            filter_spec=_deserialise_leaf,
                        )
                    except Exception as error:
                        raise ValueError(
                            "The Zodiax JAX array payload could not be decoded."
                        ) from error
                    if payload.read(1) != b"":
                        raise ValueError("The Zodiax payload contains unconsumed data.")

                # Confirm that decoding did not alter the declared object definition.
                definition.validate(loaded)
                _validate_modules(loaded)
                validate_links(loaded)
                return loaded
        except zipfile.BadZipFile as error:
            raise ValueError("The input is not a valid Zodiax ZIP archive.") from error
