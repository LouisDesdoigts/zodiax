"""Global numerical unit conventions and composable unit transforms.

Zodiax deliberately keeps realised values as ordinary JAX arrays. The active unit
system therefore defines the numerical convention used by those arrays, while a
:class:`Unit` transform converts a stored coordinate from its declared unit into the
active convention captured when the transform is constructed.

The public unit system is a small mapping such as
``{"cartesian": "um", "angular": "mas"}``. Conversion definitions are an
internal catalogue rather than a mutable user registry.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import math
from threading import RLock
from typing import Any, NamedTuple

import equinox as eqx
import jax.numpy as np
from jax import Array

from .arrays import _as_inexact_array
from .expressions import resolve
from .transforms import Transform, _operand, _validate_operand

__all__ = [
    "unit_info",
    "canonical_unit",
    "conversion_factor",
    "convert",
    "Unit",
    "get_units",
    "set_units",
]


class _UnitInfo(NamedTuple):
    """Canonical unit spelling, category, dimensional power, and reference scale."""

    unit: str
    category: str
    power: int
    factor: float


# Simple units are measured relative to metre, radian, and photon reference units.
# These references are an implementation detail; realised arrays use the active
# global unit system rather than necessarily using SI values.
_UNIT_CATALOGUE = {
    "rad": ("angular", 1.0),
    "deg": ("angular", math.pi / 180.0),
    "arcmin": ("angular", math.pi / (180.0 * 60.0)),
    "arcsec": ("angular", math.pi / (180.0 * 3_600.0)),
    "mas": ("angular", math.pi / (180.0 * 3.6e6)),
    "uas": ("angular", math.pi / (180.0 * 3.6e9)),
    "m": ("cartesian", 1.0),
    "angstrom": ("cartesian", 1e-10),
    "photon": ("photon", 1.0),
}

_UNIT_ALIASES = {
    "radian": "rad",
    "radians": "rad",
    "degree": "deg",
    "degrees": "deg",
    "am": "arcmin",
    "arcmins": "arcmin",
    "arcminute": "arcmin",
    "arcminutes": "arcmin",
    "as": "arcsec",
    "arcsecs": "arcsec",
    "arcsecond": "arcsec",
    "arcseconds": "arcsec",
    "metre": "m",
    "metres": "m",
    "meter": "m",
    "meters": "m",
    "micron": "um",
    "microns": "um",
    "µm": "um",
    "a": "angstrom",
    "aa": "angstrom",
    "ångström": "angstrom",
    "angstroms": "angstrom",
    "photons": "photon",
}

_PREFIXES = {
    "G": 1e9,
    "M": 1e6,
    "k": 1e3,
    "m": 1e-3,
    "u": 1e-6,
    "n": 1e-9,
}

_CATEGORY_ALIASES = {
    "cartesian": "cartesian",
    "length": "cartesian",
    "angular": "angular",
    "angle": "angular",
    "photon": "photon",
    "photons": "photon",
}

_DEFAULT_UNITS = {
    "cartesian": "m",
    "angular": "rad",
    "photon": "photon",
}

_LOCK = RLock()
_ACTIVE_UNITS = dict(_DEFAULT_UNITS)


def _normalise_category(category: Any) -> str:
    """Return one canonical public unit-system category."""
    if not isinstance(category, str):
        raise TypeError("Unit-system categories must be strings.")
    try:
        return _CATEGORY_ALIASES[category.strip().lower()]
    except KeyError as error:
        accepted = ", ".join(_DEFAULT_UNITS)
        raise ValueError(
            f"Unknown unit-system category {category!r}; expected {accepted}."
        ) from error


def _normalise_unit_text(unit: Any) -> str:
    """Return one bounded, non-empty unit spelling."""
    if not isinstance(unit, str):
        raise TypeError("unit must be a string.")
    unit = unit.strip()
    if not unit:
        raise ValueError("unit must not be empty.")
    if len(unit) > 128:
        raise ValueError("unit must contain at most 128 characters.")
    return unit


def _simple_unit_info(unit: str) -> _UnitInfo:
    """Resolve a non-compound unit spelling."""
    unit = _UNIT_ALIASES.get(unit.lower(), unit)
    direct = _UNIT_CATALOGUE.get(unit)
    if direct is not None:
        category, factor = direct
        return _UnitInfo(unit, category, 1, float(factor))

    if len(unit) > 1:
        prefix = unit[0]
        base = _UNIT_ALIASES.get(unit[1:].lower(), unit[1:])
        base_info = _UNIT_CATALOGUE.get(base)
        if prefix in _PREFIXES and base_info is not None:
            category, factor = base_info
            return _UnitInfo(
                prefix + base,
                category,
                1,
                float(_PREFIXES[prefix] * factor),
            )
    raise ValueError(f"Unknown unit {unit!r}.")


def _format_power(unit: str, power: int) -> str:
    """Return a canonical spelling for one powered Cartesian unit."""
    if power == 1:
        return unit
    if power == -1:
        return f"1/{unit}"
    return f"{unit}^{power}"


def unit_info(unit: str) -> _UnitInfo:
    """Return the canonical spelling, category, power, and reference factor."""
    unit = _normalise_unit_text(unit)
    return _cached_unit_info(unit)


def canonical_unit(unit: str) -> str:
    """Validate a unit and return its canonical spelling."""
    return unit_info(unit).unit


def conversion_factor(unit_in: str, unit_out: str) -> float:
    """Return the multiplicative factor converting compatible physical units."""
    source = unit_info(unit_in)
    target = unit_info(unit_out)

    if source.category != target.category or source.power != target.power:
        raise ValueError(f"Cannot convert from {unit_in!r} to {unit_out!r}.")
    return float(source.factor / target.factor)


def convert(value: Any, unit_in: str, unit_out: str) -> Any:
    """Convert a value between compatible units without global unit state."""
    return value * conversion_factor(unit_in, unit_out)


@lru_cache(maxsize=None)
def _cached_unit_info(unit: str) -> _UnitInfo:
    """Resolve one already validated spelling with an unbounded immutable cache."""

    try:
        return _simple_unit_info(unit)
    except ValueError:
        pass

    reciprocal = unit.startswith("1/")
    body = unit[2:] if reciprocal else unit
    if "^" in body:
        if body.count("^") != 1:
            raise ValueError(f"Unknown unit {unit!r}.")
        body, exponent_text = body.split("^", 1)
        try:
            power = int(exponent_text)
        except ValueError as error:
            raise ValueError(f"Unknown unit {unit!r}.") from error
    else:
        power = 1
    if reciprocal:
        power = -power
    if power == 0 or abs(power) > 8:
        raise ValueError("Unit powers must be nonzero integers between -8 and 8.")

    base = _simple_unit_info(body)
    if base.category != "cartesian":
        raise ValueError("Only Cartesian units currently support derived powers.")
    return _UnitInfo(
        _format_power(base.unit, power),
        base.category,
        power,
        float(base.factor**power),
    )


def _target_info(source: _UnitInfo) -> _UnitInfo:
    """Return the active realised unit matching one source definition."""
    with _LOCK:
        target_unit = _ACTIVE_UNITS[source.category]
    target = unit_info(target_unit)
    if source.power == 1:
        return target
    return _UnitInfo(
        _format_power(target.unit, source.power),
        source.category,
        source.power,
        float(target.factor**source.power),
    )


def get_units() -> dict[str, str]:
    """Return an independent snapshot of the active global unit system.

    The mapping defines the numerical units used by subsequently constructed
    :class:`Unit` transforms whose ``to`` argument is omitted. Mutating the returned
    dictionary has no effect on the active system.
    """
    with _LOCK:
        return dict(_ACTIVE_UNITS)


def set_units(units: Mapping[str, str]) -> dict[str, str]:
    """Atomically replace the global unit system and return its normalised value.

    ``units`` may override any of ``"cartesian"``, ``"angular"``, or
    ``"photon"``. Omitted categories retain their built-in defaults. Category
    aliases ``"length"`` and ``"angle"`` are accepted. Every chosen unit must be
    a simple unit of the corresponding category.

    Existing :class:`Unit` objects retain their stored output unit; this setting only
    affects subsequently constructed objects whose ``to`` argument is omitted.
    """
    global _ACTIVE_UNITS

    if not isinstance(units, Mapping):
        raise TypeError("units must be a mapping of category names to unit strings.")

    normalised = dict(_DEFAULT_UNITS)
    supplied: set[str] = set()
    for raw_category, raw_unit in units.items():
        category = _normalise_category(raw_category)
        if category in supplied:
            raise ValueError(f"Unit-system category {category!r} was supplied twice.")
        supplied.add(category)

        info = unit_info(raw_unit)
        if info.category != category or info.power != 1:
            raise ValueError(f"Unit {raw_unit!r} is not a simple {category!r} unit.")
        normalised[category] = info.unit

    with _LOCK:
        _ACTIVE_UNITS = normalised
        return dict(_ACTIVE_UNITS)


class Unit(Transform):
    """Convert a stored coordinate into an explicit or global numerical unit.

    Parameters
    ----------
    unit : str
        Unit of the stored or explicitly supplied input coordinate.
    x : array-like, Expression, or None
        Stored input coordinate. Leave as ``None`` to construct an unbound symbolic
        transform.
    to : str or None
        Output unit. When omitted, the active global unit for ``unit``'s physical
        category is captured at construction.
    alias : alias specification, optional
        Local aliases inherited from :class:`~zodiax.base.Module`.

    Notes
    -----
    The selected target is stored in ``to``. Changing the global system later never
    changes an existing transform's meaning.
    """

    unit: str = eqx.field(static=True)
    to: str = eqx.field(static=True)
    _scale: float = eqx.field(static=True, repr=False)

    def __init__(
        self,
        unit: str,
        x: Any = None,
        *,
        to: str | None = None,
        alias: Any = None,
    ):
        source = unit_info(unit)
        target = _target_info(source) if to is None else unit_info(to)
        if source.category != target.category or source.power != target.power:
            raise ValueError(f"Cannot convert from {unit!r} to {to!r}.")

        self.alias = alias
        self.x = _operand(x, "x", optional=True)
        self.unit = source.unit
        self.to = target.unit
        self._scale = float(source.factor / target.factor)

    @property
    def category(self) -> str:
        """Physical category shared by the input and realised units."""
        return unit_info(self.unit).category

    @property
    def factor(self) -> float:
        """Frozen multiplicative factor from ``unit`` to ``to``."""
        return self._scale

    def _convert(self, value: Any, name: str, *, inverse: bool) -> Array:
        """Apply the frozen scale without silently losing it in a narrow dtype."""
        value = _as_inexact_array(value, name)
        dtype = np.result_type(value.dtype, np.float32)
        value = np.asarray(value, dtype=dtype)
        scale = np.asarray(self._scale, dtype=dtype)
        factor = np.reciprocal(scale) if inverse else scale
        source, target = (self.to, self.unit) if inverse else (self.unit, self.to)
        factor = eqx.error_if(
            factor,
            np.logical_or(~np.isfinite(factor), factor == 0),
            f"Unit conversion {source!r} -> {target!r} is not representable "
            f"in dtype {dtype}.",
        )
        return value * factor

    def fwd(self, x: Any, **context: Any) -> Array:
        """Convert an explicitly supplied input coordinate into ``to`` units."""
        return self._convert(resolve(x, **context), "x", inverse=False)

    def inv(self, y: Any, **context: Any) -> Array:
        """Convert a realised value back into the stored coordinate unit."""
        return self._convert(resolve(y, **context), "y", inverse=True)

    def __zodiax_validate__(self) -> None:
        """Validate constructor-free archive reconstruction."""
        _validate_operand(self.x, "x", optional=True)
        source = unit_info(self.unit)
        target = unit_info(self.to)
        if source.unit != self.unit or target.unit != self.to:
            raise ValueError("unit and to must use canonical spellings.")
        if source.category != target.category or source.power != target.power:
            raise ValueError("unit and to must have compatible physical dimensions.")
        expected = float(source.factor / target.factor)
        if type(self._scale) is not float or not math.isfinite(self._scale):
            raise TypeError("_scale must be a finite Python float.")
        if self._scale == 0 or self._scale != expected:
            raise ValueError("_scale does not match the stored unit conversion.")
