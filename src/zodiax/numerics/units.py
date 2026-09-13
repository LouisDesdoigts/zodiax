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
import math
from typing import Any, NamedTuple

import equinox as eqx
import jax.numpy as np
from jax import Array

from .transforms import Transform, _operand

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

_ACTIVE_UNITS = dict(_DEFAULT_UNITS)
"""Default units for new transforms. set_units replaces this dictionary after
validation; get_units returns a copy. Existing Unit objects keep their captured
unit names, so later changes here do not alter their calculations.
"""


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


def _simple_unit_info(unit: str) -> _UnitInfo:
    """Resolve a non-compound unit spelling."""
    unit = _UNIT_ALIASES.get(unit.lower(), unit)
    direct = _UNIT_CATALOGUE.get(unit)
    if direct is not None:
        category, factor = direct
        return _UnitInfo(unit, category, 1, factor)

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
                _PREFIXES[prefix] * factor,
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
    """Return the canonical spelling, category, power, and reference factor.

    Factors refer to metres, radians, or photons, independently of ``set_units``.
    Cartesian units also support reciprocals and integer powers, such as ``1/mm``
    and ``um^2``. Prefixes are case-sensitive: ``mm`` and ``Mm`` differ.
    """
    if not isinstance(unit, str):
        raise TypeError("unit must be a string.")
    unit = unit.strip()
    if not unit:
        raise ValueError("unit must not be empty.")

    # Separate the reciprocal and exponent from the base unit spelling.
    reciprocal = unit.startswith("1/")
    body = unit[2:] if reciprocal else unit
    if not reciprocal and "^" not in body:
        return _simple_unit_info(body)

    if "^" in body:
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

    # Apply the power to both the canonical name and its reference factor.
    base = _simple_unit_info(body)
    if base.category != "cartesian":
        raise ValueError("Only Cartesian units currently support derived powers.")
    return _UnitInfo(
        _format_power(base.unit, power),
        base.category,
        power,
        base.factor**power,
    )


def canonical_unit(unit: str) -> str:
    """Validate a unit and return its canonical spelling."""
    return unit_info(unit).unit


def conversion_factor(unit_in: str, unit_out: str) -> float:
    """Return the multiplicative factor converting compatible physical units."""
    source = unit_info(unit_in)
    target = unit_info(unit_out)

    if source.category != target.category or source.power != target.power:
        raise ValueError(f"Cannot convert from {unit_in!r} to {unit_out!r}.")
    return source.factor / target.factor


def convert(value: Any, unit_in: str, unit_out: str) -> Any:
    """Convert a scalar or array between compatible units without global state.

    Float16 and bfloat16 inputs are promoted to float32 before scaling. Results
    remain Python scalars, NumPy values, or JAX arrays according to the input.
    """
    factor = conversion_factor(unit_in, unit_out)
    if eqx.is_array(value) and value.dtype in (np.float16, np.bfloat16):
        value = value.astype(np.float32)
    return value * factor


def get_units() -> dict[str, str]:
    """Return an independent snapshot of the active global unit system.

    The mapping defines the numerical units used by subsequently constructed
    :class:`Unit` transforms whose ``to`` argument is omitted. Mutating the returned
    dictionary has no effect on the active system.
    """
    return dict(_ACTIVE_UNITS)


def set_units(units: Mapping[str, str]) -> dict[str, str]:
    """Replace the global unit system and return an independent normalised copy.

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

    _ACTIVE_UNITS = normalised
    return dict(normalised)


class Unit(Transform):
    """Convert a stored coordinate into an explicit or global numerical unit.

    Parameters
    ----------
    x : array-like, Expression, or None
        Stored input coordinate. Leave as ``None`` to construct an unbound symbolic
        transform.
    unit : str
        Unit of the stored or explicitly supplied input coordinate.
    to : str or None
        Output unit. When omitted, the active global unit for ``unit``'s physical
        category is captured at construction.
    alias : alias specification, optional
        Local aliases inherited from :class:`~zodiax.base.Module`.

    Notes
    -----
    The selected target is stored in ``output_unit``. Use ``to(unit)`` to create
    a transform with a different output unit. Changing the global system later
    never changes an existing transform's meaning. Conversion arithmetic uses at
    least float32 precision; scales outside the resulting dtype's range follow
    ordinary JAX overflow and underflow behaviour.
    """

    unit: str = eqx.field(static=True)
    output_unit: str = eqx.field(static=True)

    def __init__(
        self,
        x: Any = None,
        unit: str | None = None,
        *,
        to: str | None = None,
        alias: Any = None,
    ):
        """Initialise a fixed unit conversion and its optional stored input."""
        source = unit_info(unit)
        if to is None:
            target_unit = get_units()[source.category]
            to = _format_power(target_unit, source.power)
        target = unit_info(to)
        if source.category != target.category or source.power != target.power:
            raise ValueError(f"Cannot convert from {unit!r} to {to!r}.")

        self.alias = alias
        self.x = _operand(x, "x", optional=True)
        self.unit = source.unit
        self.output_unit = target.unit

    def to(self, unit: str) -> Unit:
        """Return a new Unit with the requested output unit.

        Keep the stored input, its unit, and aliases. The constructor checks that
        the requested unit has compatible dimensions; no input is evaluated.
        For example, ``Unit(1.0, "m").to("mm").resolve()`` returns ``1000.0``.
        """
        return Unit(self.x, self.unit, to=unit, alias=self.alias)

    @property
    def category(self) -> str:
        """Physical category shared by the input and realised units."""
        return unit_info(self.unit).category

    @property
    def factor(self) -> float:
        """Multiplicative factor derived from the two stored unit names.

        Both names are static, so JIT calculates this Python float during tracing.
        """
        return conversion_factor(self.unit, self.output_unit)

    def fwd(self, x: Array, **context: Any) -> Array:
        """Convert a prepared input array into ``output_unit`` units."""
        # Even ordinary unit factors can exceed the range of float16.
        dtype = np.result_type(x.dtype, np.float32)
        return x.astype(dtype) * self.factor

    def inv(self, y: Array, **context: Any) -> Array:
        """Convert a prepared output array back into the stored coordinate unit."""
        dtype = np.result_type(y.dtype, np.float32)
        return y.astype(dtype) / self.factor
