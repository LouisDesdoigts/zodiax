"""Unit conversion and global settings: overview developer contract 12.4."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import zodiax as zdx


@pytest.fixture(autouse=True)
def unit_settings():
    previous = zdx.get_units()
    zdx.set_units({})
    yield
    zdx.set_units(previous)


def test_to_returns_an_array_without_replacing_the_stored_definition():
    coefficients = zdx.Map(x=[1.0, 2.0], s=2.0)
    distance = zdx.Unit(x=coefficients, unit="m", alias={"latent": "x.x"})

    millimetres = distance.to("mm")

    assert isinstance(millimetres, jax.Array)
    np.testing.assert_allclose(millimetres, [2000.0, 4000.0])
    assert distance.x is coefficients
    assert distance.unit_out is None
    np.testing.assert_allclose(distance.latent, [1.0, 2.0])
    np.testing.assert_allclose(distance.resolve(), [2.0, 4.0])


def test_to_converts_the_stored_input_once_and_uses_the_requested_target():
    distance = zdx.Unit(x=zdx.Map(x=2.0, s=3.0), unit="mm", to="m")

    np.testing.assert_allclose(distance.resolve(), 0.006)
    np.testing.assert_allclose(distance.to("um"), 6000.0)
    np.testing.assert_allclose(distance.to("mm"), 6.0)
    assert distance.unit_out == "m"


def test_to_accepts_runtime_context_for_a_deferred_input():
    distance = zdx.Unit(x=zdx.Map(x=zdx.StateRef("distance"), s=2.0), unit="m")
    state = zdx.State(distance=3.0)

    value = distance.to("mm", state=state)

    assert isinstance(value, jax.Array)
    np.testing.assert_allclose(value, 6000.0)
    assert isinstance(distance.x.x, zdx.StateRef)


@pytest.mark.parametrize("input", [None, zdx.StateRef("distance")])
def test_to_requires_enough_information_to_return_an_array(input):
    distance = zdx.Unit(x=input, unit="m")

    with pytest.raises(ValueError):
        distance.to("mm")
    assert isinstance(distance.resolve(), zdx.Unit)


def test_default_resolution_and_to_none_use_current_global_settings():
    following = zdx.Unit(x=2.0, unit="mm")
    fixed = zdx.Unit(x=2.0, unit="mm", to="m")
    np.testing.assert_allclose(following.resolve(), 0.002)

    zdx.set_units({"cartesian": "um"})

    np.testing.assert_allclose(following.resolve(), 2000.0)
    np.testing.assert_allclose(fixed.resolve(), 0.002)
    np.testing.assert_allclose(fixed.to(None), 2000.0)
    assert following.unit_out is None
    assert fixed.unit_out == "m"


@pytest.mark.parametrize(
    ("unit", "metres", "micrometres"),
    [("mm^2", 2e-6, 2e6), ("1/mm", 2000.0, 0.002)],
)
def test_global_conventions_preserve_dimensional_powers(unit, metres, micrometres):
    value = zdx.Unit(x=2.0, unit=unit)
    np.testing.assert_allclose(value.to(None), metres)

    zdx.set_units({"cartesian": "um"})

    np.testing.assert_allclose(value.to(None), micrometres)


def test_jit_keeps_the_global_convention_until_retracing():
    distance = zdx.Unit(x=2.0, unit="mm")
    compiled = eqx.filter_jit(lambda model: model.to(None))
    np.testing.assert_allclose(compiled(distance), 0.002)

    zdx.set_units({"cartesian": "um"})

    np.testing.assert_allclose(distance.to(None), 2000.0)
    np.testing.assert_allclose(compiled(distance), 0.002)
    # A new input shape forces a trace that reads the updated setting.
    vector = distance.set(x=zdx.as_array([2.0, 3.0]))
    np.testing.assert_allclose(compiled(vector), [2000.0, 3000.0])


def test_to_supports_compilation_gradients_and_vectorised_context():
    distance = zdx.Unit(x=zdx.Map(x=2.0, s=3.0), unit="m")
    compiled = eqx.filter_jit(lambda model: model.to("mm"))
    np.testing.assert_allclose(compiled(distance), 6000.0)

    gradient = jax.grad(lambda model: model.to("mm"))(distance)
    np.testing.assert_allclose(gradient.x.x, 3000.0)
    np.testing.assert_allclose(gradient.x.s, 2000.0)

    contextual = zdx.Unit(x=zdx.StateRef("distance"), unit="m")
    values = jax.vmap(
        lambda value: contextual.to("mm", state=zdx.State(distance=value))
    )(jnp.array([1.0, 2.0]))
    np.testing.assert_allclose(values, [1000.0, 2000.0])


def test_local_unit_scaling_retains_complex_values_and_has_a_reverse():
    conversion = zdx.Unit(unit="m", to="mm")
    value = jnp.array(1 + 2j)
    output = conversion.apply(value)

    np.testing.assert_allclose(output, 1000 + 2000j)
    np.testing.assert_allclose(conversion.apply(output, inverse=True), value)
    np.testing.assert_allclose(zdx.Unit(x=value, unit="m").to("mm"), output)


@pytest.mark.parametrize(
    ("source", "target", "factor"),
    [
        ("m", "mm", 1000.0),
        ("mm^2", "m^2", 1e-6),
        ("1/mm", "1/m", 1000.0),
        ("deg", "rad", np.pi / 180),
        ("mas", "arcsec", 0.001),
        ("kphoton", "photon", 1000.0),
    ],
)
def test_explicit_conversion_matches_known_physical_factors(source, target, factor):
    assert zdx.conversion_factor(source, target) == pytest.approx(factor)
    np.testing.assert_allclose(zdx.Unit(x=2.0, unit=source).to(target), 2 * factor)


def test_unit_aliases_preserve_case_sensitive_prefixes():
    assert zdx.canonical_unit("microns") == "um"
    assert zdx.canonical_unit("µm") == "um"
    assert zdx.unit_info("mm^2").power == 2
    assert zdx.conversion_factor("Mm", "m") == pytest.approx(1e6)
    assert zdx.conversion_factor("mm", "m") == pytest.approx(1e-3)


def test_convert_uses_ordinary_multiplication_without_forcing_an_array():
    scalar = zdx.convert(2.0, "m", "mm")
    array = zdx.convert(jnp.array([1.0, 2.0]), "m", "mm")

    assert isinstance(scalar, float)
    assert scalar == pytest.approx(2000.0)
    np.testing.assert_allclose(array, [1000.0, 2000.0])


def test_settings_are_independent_copies_and_invalid_changes_are_atomic():
    selected = zdx.set_units({"length": "mm"})
    snapshot = zdx.get_units()
    selected["cartesian"] = "m"
    snapshot["cartesian"] = "um"
    assert zdx.get_units()["cartesian"] == "mm"
    assert zdx.get_units()["angular"] == "rad"

    with pytest.raises(ValueError):
        zdx.set_units({"cartesian": "um", "angular": "m"})

    assert zdx.get_units()["cartesian"] == "mm"


@pytest.mark.parametrize("target", ["rad", "m^2"])
def test_unit_conversion_requires_compatible_dimensions(target):
    with pytest.raises(ValueError):
        zdx.Unit(x=2.0, unit="m").to(target)


def test_default_unit_target_is_omitted_from_normal_equinox_printing():
    following = zdx.Unit(x=2.0, unit="mm")
    fixed = zdx.Unit(x=2.0, unit="mm", to="m")

    assert "unit_out=" not in repr(following)
    assert "unit_out='m'" in repr(fixed)
