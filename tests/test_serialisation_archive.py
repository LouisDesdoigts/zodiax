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
    ("mutate", "message"),
    [
        (lambda manifest: manifest.pop("definition"), "missing required fields"),
        (
            lambda manifest: manifest.__setitem__("format", {}),
            "not a supported Zodiax archive",
        ),
        (
            lambda manifest: manifest["format"].__setitem__("version", 2),
            "Unsupported Zodiax archive version",
        ),
        (
            lambda manifest: manifest.__setitem__("definition", []),
            "Object definition node.*must be a mapping",
        ),
        (
            lambda manifest: manifest.__setitem__("created_with", []),
            "invalid provenance",
        ),
        (
            lambda manifest: manifest["created_with"].__setitem__("packages", []),
            "invalid package provenance",
        ),
        (
            lambda manifest: manifest["created_with"]["packages"].__setitem__("jax", 1),
            "invalid version for package 'jax'",
        ),
        (
            lambda manifest: manifest["created_with"]["packages"].__setitem__(
                "jax", ""
            ),
            "invalid version for package 'jax'",
        ),
        (
            lambda manifest: manifest.__setitem__("payload", []),
            "invalid payload description",
        ),
        (
            lambda manifest: manifest["payload"].__setitem__("member", "other"),
            "unsupported payload member",
        ),
        (
            lambda manifest: manifest["payload"].__setitem__("codec", "other"),
            "Unsupported Zodiax payload codec",
        ),
        (
            lambda manifest: manifest["payload"].__setitem__("version", True),
            "Unsupported Zodiax payload version",
        ),
        (
            lambda manifest: manifest["payload"].__setitem__("size", -1),
            "payload size must be a non-negative integer",
        ),
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
def test_load_rejects_invalid_manifest(mutate, message):
    file = BytesIO()
    zdx.save(file, _original())

    _rewrite_archive(file, update_manifest=lambda manifest, payload: mutate(manifest))

    with pytest.raises(ValueError, match=message):
        zdx.load(file)


def test_definition_schema_is_validated_before_types_or_payload():
    file = BytesIO()
    zdx.save(file, _original())

    def mutate(manifest, payload):
        manifest["definition"]["type"] = "missing.module:Ghost"
        manifest["definition"]["fields"][0]["value"]["shape"] = [True]
        manifest["payload"]["size"] = len(payload)

    _rewrite_archive(file, update_manifest=mutate, update_payload=lambda payload: b"x")

    with pytest.raises(ValueError, match="Invalid JAX array shape"):
        zdx.load(file)


@pytest.mark.parametrize(
    ("archive_change", "message"),
    [
        ("duplicate-member", "duplicate members"),
        ("extra-member", "must contain manifest.json and leaves.eqx"),
        ("compressed", "Compressed Zodiax archive members are unsupported"),
        ("duplicate-json", "manifest is not valid UTF-8 JSON"),
    ],
)
def test_load_rejects_ambiguous_archive_framing(archive_change, message):
    file = BytesIO()
    zdx.save(file, _original())
    manifest, payload = _read_archive(file)

    if archive_change == "duplicate-member":
        with pytest.warns(UserWarning, match="Duplicate name"):
            _write_archive(file, manifest, payload, duplicate_manifest=True)
    elif archive_change == "extra-member":
        _write_archive(file, manifest, payload, extra_member="extra.bin")
    elif archive_change == "compressed":
        _write_archive(file, manifest, payload, compression=zipfile.ZIP_DEFLATED)
    else:
        encoded = json.dumps(manifest)[1:]
        raw_manifest = ('{"definition": null,' + encoded).encode()
        _write_archive(file, manifest, payload, raw_manifest=raw_manifest)

    with pytest.raises(ValueError, match=message):
        zdx.load(file)


def test_load_rejects_payload_size_mismatch():
    file = BytesIO()
    zdx.save(file, np.ones(2))

    def mutate(manifest, payload):
        del payload
        manifest["payload"]["size"] += 1

    _rewrite_archive(file, update_manifest=mutate)

    with pytest.raises(ValueError, match="payload size does not match"):
        zdx.load(file)


def test_load_requires_recorded_package_versions_by_default():
    file = BytesIO()
    original = _original()
    zdx.save(file, original)

    def mutate(manifest, payload):
        del payload
        manifest["created_with"]["packages"]["jax"] = "incompatible-version"

    _rewrite_archive(file, update_manifest=mutate)

    with pytest.raises(ValueError, match="package version mismatch.*jax"):
        zdx.load(file)

    loaded = zdx.load(file, strict=False)
    assert eqx.tree_equal(loaded, original, typematch=True)


def test_load_strict_must_be_boolean():
    file = BytesIO()
    zdx.save(file, _original())

    with pytest.raises(TypeError, match="strict must be a bool"):
        zdx.load(file, strict=1)


def test_load_rejects_corrupt_payload():
    file = BytesIO()
    zdx.save(file, np.ones(2))

    def corrupt(payload):
        return b"BAD" + payload[3:]

    _rewrite_archive(file, update_payload=corrupt)

    with pytest.raises(ValueError, match="payload could not be decoded"):
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

    with pytest.raises(ValueError, match="payload contains unconsumed data"):
        zdx.load(file)


def test_load_rejects_non_archive():
    with pytest.raises(ValueError, match="not a valid Zodiax ZIP archive"):
        zdx.load(BytesIO(b"not an archive"))
