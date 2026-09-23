"""Overview §12.4: numerical state, contextual references, and pure transitions."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import zodiax as zdx


class NeedsState(zdx.Expression):
    """State validation inspects references without running this calculation."""

    current: zdx.StateRef

    def evaluate(self, **context):
        raise RuntimeError("numerical calculation failed")


def test_constructor_stores_arrays_with_inferred_types_and_preserves_typed_keys():
    key = jr.key(17)
    state = zdx.State(index=0, time=1.0, samples=[2.0, 5.0], amplitude=1 + 2j, key=key)

    assert all(isinstance(value, jax.Array) for value in state.values.values())
    assert jnp.issubdtype(state.value("index").dtype, jnp.integer)
    assert state.value("index").shape == ()
    assert state.value("samples").shape == (2,)
    np.testing.assert_allclose(state.value("amplitude"), 1 + 2j)
    assert state.value("key").dtype == key.dtype
    np.testing.assert_array_equal(jr.key_data(state.value("key")), jr.key_data(key))


def test_constructor_keyword_entries_override_mapping_entries():
    state = zdx.State({"time": 1.0, "index": 0}, time=2.0)

    np.testing.assert_allclose(state.value("time"), 2.0)
    np.testing.assert_array_equal(state.value("index"), 0)


@pytest.mark.parametrize("value", [None, zdx.Map(x=1.0, s=2.0)])
def test_state_requires_current_numbers_rather_than_missing_values_or_definitions(
    value,
):
    with pytest.raises(TypeError):
        zdx.State(current=value)


def test_references_defer_without_state_and_read_each_supplied_state():
    state = zdx.State(time=1.0)
    reference = state.ref("time")
    assert isinstance(reference.resolve(), zdx.StateRef)
    definition = zdx.Map(x=reference, b=2.0)

    np.testing.assert_allclose(definition.resolve(state=state), 3.0)
    changed = state.set(time=zdx.as_array(4.0))
    np.testing.assert_allclose(definition.resolve(state=changed), 6.0)
    np.testing.assert_allclose(state.value("time"), 1.0)


def test_qualified_state_paths_disambiguate_entry_names_from_methods():
    state = zdx.State(step=3.0, time=1.0, alias={"clock": "values.time"})

    np.testing.assert_allclose(state.ref("values.step").resolve(state=state), 3.0)
    np.testing.assert_allclose(zdx.StateRef("clock").resolve(state=state), 1.0)


def test_missing_supplied_state_paths_raise_instead_of_deferring():
    state = zdx.State(time=1.0)

    with pytest.raises(KeyError):
        zdx.StateRef("missing").resolve(state=state)
    with pytest.raises(KeyError):
        state.ref("missing")


def test_inherited_set_keeps_conversion_and_carry_structure_with_the_caller():
    state = zdx.State(samples=[1.0, 2.0])
    replacement = [3.0, 4.0, 5.0]

    changed = state.set(samples=replacement)

    assert isinstance(changed.value("samples"), list)
    assert changed.value("samples") == replacement
    np.testing.assert_allclose(state.value("samples"), [1.0, 2.0])


def test_array_index_supports_compiled_gather_and_indexed_updates():
    state = zdx.State(index=1, samples=[2.0, 5.0, 7.0])

    @eqx.filter_jit
    def sample_and_update(current):
        index = current.value("index")
        values = current.value("samples")
        updated = current.set(samples=values.at[index].set(11.0))
        return values[index], updated

    sampled, updated = sample_and_update(state)
    np.testing.assert_allclose(sampled, 5.0)
    np.testing.assert_allclose(updated.value("samples"), [2.0, 11.0, 7.0])
    sampled, updated = sample_and_update(state.set(index=zdx.as_array(2)))
    np.testing.assert_allclose(sampled, 7.0)
    np.testing.assert_allclose(updated.value("samples"), [2.0, 5.0, 11.0])
    np.testing.assert_allclose(state.value("samples"), [2.0, 5.0, 7.0])


def test_array_index_supports_fixed_size_dynamic_slices():
    state = zdx.State(index=1, samples=[2.0, 5.0, 7.0, 11.0])
    compiled = eqx.filter_jit(
        lambda current: jax.lax.dynamic_slice_in_dim(
            current.value("samples"), current.value("index"), 2
        )
    )

    np.testing.assert_allclose(compiled(state), [5.0, 7.0])
    np.testing.assert_allclose(compiled(state.set(index=zdx.as_array(2))), [7.0, 11.0])


def test_state_can_be_constructed_from_traced_values():
    @jax.jit
    def make_state(index, time):
        return zdx.State(index=index, time=time)

    state = make_state(jnp.array(2), jnp.array(3.0))

    np.testing.assert_array_equal(state.value("index"), 2)
    np.testing.assert_allclose(state.value("time"), 3.0)


def test_step_is_a_scan_carry_and_leaves_unmanaged_entries_unchanged():
    initial = zdx.State(index=0, time=1.0, dt=0.25, key=jr.key(17), gain=2.0)

    def run(state):
        def advance(current, _):
            following = current.step()
            return following, following.value("time")

        return jax.lax.scan(advance, state, None, length=3)

    final, times = jax.jit(run)(initial)

    np.testing.assert_array_equal(final.value("index"), 3)
    np.testing.assert_allclose(times, [1.25, 1.5, 1.75])
    np.testing.assert_allclose(final.value("gain"), 2.0)
    np.testing.assert_array_equal(
        jr.key_data(final.value("key")), jr.key_data(initial.value("key"))
    )
    np.testing.assert_allclose(initial.value("time"), 1.0)


def test_step_reads_aliased_dt_from_the_original_index():
    state = zdx.State(index=2, time=1.0, alias={"dt": "values.index"})

    following = jax.jit(lambda current: current.step())(state)

    np.testing.assert_array_equal(following.value("index"), 3)
    np.testing.assert_allclose(following.value("time"), 3.0)


def test_explicit_step_size_overrides_the_stored_value_without_changing_time_shape():
    state = zdx.State(time=[1.0, 2.0], dt=0.25)

    np.testing.assert_allclose(state.step().value("time"), [1.25, 2.25])
    changed = state.step(dt=0.5)
    np.testing.assert_allclose(changed.value("time"), [1.5, 2.5])
    np.testing.assert_allclose(changed.value("dt"), 0.25)
    with pytest.raises(ValueError):
        state.step(dt=jnp.ones((2, 1)))


def test_index_only_state_can_step_without_time_or_dt():
    state = zdx.State(index=2)

    np.testing.assert_array_equal(state.step().value("index"), 3)


def test_step_requires_an_entry_to_advance_and_a_usable_time_step():
    with pytest.raises(ValueError):
        zdx.State().step()
    with pytest.raises(KeyError):
        zdx.State(time=1.0).step()
    with pytest.raises(ValueError):
        zdx.State(index=0).step(dt=0.5)


def test_validate_state_checks_references_without_evaluating_the_model():
    state = zdx.State(time=1.0, unrelated=2.0)
    model = NeedsState(zdx.StateRef("time"))

    assert zdx.validate_state(model, state) is None
    assert zdx.validate_state((state.ref("time"), state.ref("time")), state) is None

    with pytest.raises(ValueError):
        zdx.validate_state(NeedsState(zdx.StateRef("missing")), state)
