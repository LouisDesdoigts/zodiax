"""Regression tests for archive framing, schema validation, and payload integrity."""

from io import BytesIO
import json
import zipfile

import equinox as eqx
import jax.numpy as np
import pytest

import zodiax as zdx


class ArchiveModule(eqx.Module):
    """Small generic module used to produce a representative archive."""

    array: object
    count: int
    mode: str = eqx.field(static=True)


def _original():
    return ArchiveModule(np.ones((2, 2)), 2, "science")


def _read_archive(file):
    """Return the decoded manifest and raw payload from an in-memory archive."""
    file.seek(0)
    with zipfile.ZipFile(file) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        payload = archive.read("leaves.eqx")
    return manifest, payload


def _write_archive(
    file,
    manifest,
    payload,
    *,
    compression=zipfile.ZIP_STORED,
    raw_manifest=None,
    extra_member=None,
    duplicate_manifest=False,
):
    """Replace an in-memory archive with selected valid ZIP framing."""
    data = json.dumps(manifest).encode() if raw_manifest is None else raw_manifest
    rebuilt = BytesIO()
    with zipfile.ZipFile(rebuilt, "w", compression=compression) as archive:
        archive.writestr("manifest.json", data)
        if duplicate_manifest:
            archive.writestr("manifest.json", data)
        archive.writestr("leaves.eqx", payload)
        if extra_member is not None:
            archive.writestr(extra_member, b"extra")
    file.seek(0)
    file.truncate(0)
    file.write(rebuilt.getvalue())
    file.seek(0)


def _rewrite_archive(file, *, update_manifest=None, update_payload=None):
    """Rebuild an archive after applying selected mutations."""
    manifest, payload = _read_archive(file)
    if update_payload is not None:
        payload = update_payload(payload)
    if update_manifest is not None:
        update_manifest(manifest, payload)
    _write_archive(file, manifest, payload)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda manifest: manifest.pop("definition"),
        lambda manifest: manifest.__setitem__("format", {}),
        lambda manifest: manifest["format"].__setitem__("version", 2),
        lambda manifest: manifest.__setitem__("definition", []),
        lambda manifest: manifest.__setitem__("created_with", []),
        lambda manifest: manifest["created_with"].__setitem__("packages", []),
        lambda manifest: manifest["created_with"]["packages"].__setitem__("jax", 1),
        lambda manifest: manifest["created_with"]["packages"].__setitem__("jax", ""),
        lambda manifest: manifest.__setitem__("payload", []),
        lambda manifest: manifest["payload"].__setitem__("member", "other"),
        lambda manifest: manifest["payload"].__setitem__("codec", "other"),
        lambda manifest: manifest["payload"].__setitem__("version", True),
        lambda manifest: manifest["payload"].__setitem__("size", -1),
    ],
    ids=[
        "missing-definition",
        "format",
        "format-version",
        "definition",
        "provenance",
        "provenance-packages",
        "provenance-version",
        "empty-provenance-version",
        "payload",
        "payload-member",
        "payload-codec",
        "payload-version",
        "payload-size",
    ],
)
def test_load_rejects_invalid_manifest(mutate):
    file = BytesIO()
    zdx.save(file, _original())

    _rewrite_archive(file, update_manifest=lambda manifest, payload: mutate(manifest))

    with pytest.raises(ValueError):
        zdx.load(file)


def test_definition_schema_is_validated_before_types_or_payload():
    file = BytesIO()
    zdx.save(file, _original())

    def mutate(manifest, payload):
        manifest["definition"]["type"] = "missing.module:Ghost"
        manifest["definition"]["fields"][0]["value"]["shape"] = [True]
        manifest["payload"]["size"] = len(payload)

    _rewrite_archive(file, update_manifest=mutate, update_payload=lambda payload: b"x")

    with pytest.raises(ValueError, match="root.array"):
        zdx.load(file)


@pytest.mark.parametrize(
    "archive_change",
    [
        "duplicate-member",
        "extra-member",
        "compressed",
        "duplicate-json",
    ],
)
def test_load_rejects_ambiguous_archive_framing(archive_change):
    file = BytesIO()
    zdx.save(file, _original())
    manifest, payload = _read_archive(file)

    if archive_change == "duplicate-member":
        with pytest.warns(UserWarning):
            _write_archive(file, manifest, payload, duplicate_manifest=True)
    elif archive_change == "extra-member":
        _write_archive(file, manifest, payload, extra_member="extra.bin")
    elif archive_change == "compressed":
        _write_archive(file, manifest, payload, compression=zipfile.ZIP_DEFLATED)
    else:
        encoded = json.dumps(manifest)[1:]
        raw_manifest = ('{"definition": null,' + encoded).encode()
        _write_archive(file, manifest, payload, raw_manifest=raw_manifest)

    with pytest.raises(ValueError):
        zdx.load(file)


def test_load_rejects_payload_size_mismatch():
    file = BytesIO()
    zdx.save(file, np.ones(2))

    def mutate(manifest, payload):
        manifest["payload"]["size"] += 1

    _rewrite_archive(file, update_manifest=mutate)

    with pytest.raises(ValueError):
        zdx.load(file)


def test_load_requires_recorded_package_versions_by_default():
    file = BytesIO()
    original = _original()
    zdx.save(file, original)

    def mutate(manifest, payload):
        manifest["created_with"]["packages"]["jax"] = "incompatible-version"

    _rewrite_archive(file, update_manifest=mutate)

    with pytest.raises(ValueError, match="jax"):
        zdx.load(file)

    loaded = zdx.load(file, strict=False)
    assert eqx.tree_equal(loaded, original, typematch=True)


def test_load_strict_must_be_boolean():
    file = BytesIO()
    zdx.save(file, _original())

    with pytest.raises(TypeError):
        zdx.load(file, strict=1)


def test_load_rejects_corrupt_payload():
    file = BytesIO()
    zdx.save(file, np.ones(2))

    def corrupt(payload):
        return b"BAD" + payload[3:]

    _rewrite_archive(file, update_payload=corrupt)

    with pytest.raises(ValueError):
        zdx.load(file)


def test_load_rejects_unconsumed_payload():
    file = BytesIO()
    zdx.save(file, np.ones(2))

    def update_size(manifest, payload):
        manifest["payload"]["size"] = len(payload)

    _rewrite_archive(
        file,
        update_manifest=update_size,
        update_payload=lambda payload: payload + b"EXTRA",
    )

    with pytest.raises(ValueError):
        zdx.load(file)


def test_load_rejects_non_archive():
    with pytest.raises(ValueError):
        zdx.load(BytesIO(b"not an archive"))


@pytest.mark.parametrize("version", [(1, 0), (2, 0)])
def test_fortran_array_payload_preserves_values(version):
    import numpy as onp

    original = np.arange(6, dtype=np.float32).reshape(2, 3)
    file = BytesIO()
    zdx.save(file, original)
    payload = BytesIO()
    onp.lib.format.write_array(
        payload, onp.asfortranarray(onp.asarray(original)), version=version
    )
    manifest, _ = _read_archive(file)
    manifest["payload"]["size"] = len(payload.getvalue())
    _write_archive(file, manifest, payload.getvalue())

    assert np.array_equal(zdx.load(file), original)


@pytest.mark.parametrize("corruption", ["shape", "dtype", "truncated", "npy-version"])
def test_array_payload_must_match_definition(corruption):
    import numpy as onp

    file = BytesIO()
    zdx.save(file, np.ones(2, dtype=np.float32))
    manifest, original_payload = _read_archive(file)
    payload = BytesIO()
    if corruption == "shape":
        onp.save(payload, onp.ones(3, dtype=onp.float32))
    elif corruption == "dtype":
        onp.save(payload, onp.ones(2, dtype=onp.int32))
    elif corruption == "truncated":
        payload.write(original_payload[:-1])
    else:
        payload.write(b"\x93NUMPY\x03\x00")
    manifest["payload"]["size"] = len(payload.getvalue())
    _write_archive(file, manifest, payload.getvalue())

    with pytest.raises(ValueError):
        zdx.load(file, strict=False)


def test_precision_configuration_must_preserve_archived_dtype(x64_context):
    file = BytesIO()
    with x64_context(True):
        zdx.save(file, np.asarray([1.0 + 2**-40], dtype=np.float64))
    with x64_context(False):
        with pytest.raises(ValueError):
            zdx.load(file, strict=False)
    with x64_context(True):
        loaded = zdx.load(file)
        assert loaded.dtype == np.float64
        assert loaded[0] == 1.0 + 2**-40


@pytest.mark.parametrize("destination", ["path", "stream"])
def test_encoding_failure_preserves_existing_destination(
    monkeypatch, tmp_path, destination
):
    original = np.arange(3, dtype=np.float32)
    target = tmp_path / "model.zdx" if destination == "path" else BytesIO()
    zdx.save(target, original)
    before = target.read_bytes() if destination == "path" else target.getvalue()

    def fail_encoding(file, value, **kwargs):
        raise OSError("fixture encoding failure")

    monkeypatch.setattr(eqx, "tree_serialise_leaves", fail_encoding)
    with pytest.raises(OSError):
        zdx.save(target, np.ones(2))
    after = target.read_bytes() if destination == "path" else target.getvalue()
    assert after == before
    assert sorted(path.name for path in tmp_path.iterdir()) == (
        ["model.zdx"] if destination == "path" else []
    )
    if destination == "stream":
        assert not target.closed


@pytest.mark.parametrize("progress", ["partial", "none"])
def test_stream_short_writes_raise_instead_of_reporting_success(progress):
    class ShortStream(BytesIO):
        def write(self, data):
            if progress == "none":
                return None  # Raw binary I/O signals that no bytes were accepted.
            return super().write(data[: len(data) // 2])

    destination = ShortStream()
    with pytest.raises(OSError):
        zdx.save(destination, np.ones(2))
    assert not destination.closed


def test_nonseekable_output_is_supported_and_remains_caller_owned():
    class Sink:
        def __init__(self):
            self.data = bytearray()

        def write(self, data):
            self.data.extend(data)
            return len(data)

    sink = Sink()
    zdx.save(sink, np.arange(3))
    assert np.array_equal(zdx.load(BytesIO(sink.data)), np.arange(3))


@pytest.mark.parametrize("raw_manifest", [b"\xff", b'{"definition": NaN}', b"[]"])
def test_manifest_requires_unambiguous_finite_utf8_json(raw_manifest):
    file = BytesIO()
    zdx.save(file, np.ones(1))
    manifest, payload = _read_archive(file)
    _write_archive(file, manifest, payload, raw_manifest=raw_manifest)
    with pytest.raises(ValueError):
        zdx.load(file)


def test_oversized_manifest_is_rejected_before_reconstruction():
    file = BytesIO()
    zdx.save(file, np.ones(1))
    manifest, payload = _read_archive(file)
    # This remains valid JSON and otherwise valid archive metadata. Its size alone
    # exceeds the format's bounded manifest, without a large array allocation.
    manifest["annotation"] = " " * (17 * 1024**2)
    _write_archive(file, manifest, payload)
    with pytest.raises(ValueError):
        zdx.load(file)


def test_encrypted_member_flag_is_rejected():
    import struct

    file = BytesIO()
    zdx.save(file, np.ones(1))
    data = bytearray(file.getvalue())
    # Mark the first central-directory member as encrypted. There is no password
    # handling in the format; rejection precedes reading this member's payload.
    central_header = data.index(b"PK\x01\x02")
    struct.pack_into("<H", data, central_header + 8, 1)
    with pytest.raises(ValueError):
        zdx.load(BytesIO(data))
