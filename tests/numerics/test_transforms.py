"""Overview 12.3: transform lifecycle, algebra, and promised JAX behaviour.

Small hand-solvable examples fix the forward and representative reverse results.
JIT and differentiation checks target chain order, matrix rank, batching, and
runtime mask selection rather than the implementation of private helpers.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import zodiax as zdx


def test_unbound_transforms_retain_their_definition_until_given_an_input():
    transform = zdx.Map(s=2.0, b=1.0)

    assert isinstance(transform(), zdx.Map)
    assert transform.x is None
    np.testing.assert_allclose(transform.apply([1.0, 2.0]), [3.0, 5.0])
    assert transform.x is None
    with pytest.raises(ValueError):
        transform(inverse=True)


def test_a_ready_expression_can_supply_the_explicit_deepest_input():
    transform = zdx.Map(x=zdx.Map(x=99.0, b=2.0), s=3.0)
    explicit = zdx.Map(x=1.0, s=3.0)

    np.testing.assert_allclose(transform.apply(explicit), 15.0)
    np.testing.assert_allclose(transform.x.x, 99.0)


def test_stored_calls_and_local_methods_have_distinct_transform_scopes():
    transform = zdx.Map(x=zdx.Map(x=1.0, b=2.0), s=3.0)

    np.testing.assert_allclose(transform.resolve(), 9.0)
    np.testing.assert_allclose(transform(), 9.0)
    np.testing.assert_allclose(transform.fwd(jnp.array(4.0)), 12.0)
    np.testing.assert_allclose(transform.apply(4.0), 18.0)
    np.testing.assert_allclose(transform(4.0), 18.0)
    np.testing.assert_allclose(transform.inv(jnp.array(18.0)), 6.0)
    np.testing.assert_allclose(transform.apply(18.0, inverse=True), 4.0)
    np.testing.assert_allclose(transform(18.0, inverse=True), 4.0)
    assert isinstance(zdx.Exp().resolve(), zdx.Exp)


def test_inverse_chain_undoes_noncommuting_operations_from_outermost_to_deepest():
    transform = zdx.Exp(x=zdx.Map(s=2.0, b=1.0))
    values = jnp.array([0.0, 1.0])
    outputs = jnp.array([np.e, np.e**3])

    @eqx.filter_jit
    def forward_and_inverse(model, inputs, targets):
        return model.apply(inputs), model.apply(targets, inverse=True)

    forward, recovered = forward_and_inverse(transform, values, outputs)
    np.testing.assert_allclose(forward, outputs, rtol=1e-6)
    np.testing.assert_allclose(recovered, values, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(transform(outputs, inverse=True), values, atol=1e-6)
    np.testing.assert_allclose(transform.inv(outputs), [1.0, 3.0], rtol=1e-6)


def test_initialise_binds_deepest_input_without_mutating_the_definition():
    transform = zdx.Map(x=zdx.Map(b=2.0), s=3.0)

    initialised = transform.initialise(jnp.array([9.0, 18.0]))

    assert transform.x.x is None
    np.testing.assert_allclose(initialised.x.x, [1.0, 4.0])
    np.testing.assert_allclose(initialised.resolve(), [9.0, 18.0])


def test_initialise_cannot_rebind_through_a_nontransform_expression():
    definition = zdx.Map(x=zdx.StateRef("value"), s=2.0)

    with pytest.raises(ValueError):
        definition.initialise(jnp.array(4.0))


@pytest.mark.parametrize(
    ("operands", "expected"),
    [
        ({}, [1.0, 2.0]),
        ({"s": 2.0}, [2.0, 4.0]),
        ({"b": 1.0}, [2.0, 3.0]),
        ({"M": [[2.0, 0.0], [0.0, 3.0]]}, [2.0, 6.0]),
    ],
)
def test_map_absent_operands_are_identity_in_both_directions(operands, expected):
    transform = zdx.Map(**operands)

    np.testing.assert_allclose(transform.apply([1.0, 2.0]), expected)
    np.testing.assert_allclose(
        transform.apply(expected, inverse=True), [1.0, 2.0], rtol=1e-6
    )


def test_free_matmul_contracts_the_final_input_axis_and_resolves_a_ready_matrix():
    matrix = zdx.Map(x=[[1.0, 2.0, 0.0], [0.0, -1.0, 3.0]], s=2.0)

    np.testing.assert_allclose(zdx.matmul([2.0, 5.0], matrix), [4.0, -2.0, 30.0])
    np.testing.assert_array_equal(zdx.matmul([2, 5]), [2, 5])


def test_map_order_and_batched_reverse_have_known_values():
    mapping = zdx.Map(
        s=[2.0, 0.5],
        M=[[1.0, 2.0, 0.0], [0.0, -1.0, 3.0]],
        b=[1.0, 2.0, -1.0],
    )
    inputs = jnp.array([[1.0, 2.0], [3.0, 4.0]])

    output = eqx.filter_jit(lambda model, x: model.apply(x))(mapping, inputs)

    np.testing.assert_allclose(output, [[3.0, 5.0, 2.0], [7.0, 12.0, 5.0]])
    np.testing.assert_allclose(
        mapping.apply(output, inverse=True), inputs, rtol=1e-5, atol=1e-5
    )


def test_sampled_matrix_contract_preserves_output_axes_and_input_batches():
    basis = zdx.Map(M=[[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]])
    inputs = jnp.array([[[2.0, 5.0]], [[3.0, 7.0]]])

    output = basis.apply(inputs)

    assert output.shape == (2, 1, 2, 2)
    expected = [[[[2.0, 5.0], [5.0, 2.0]]], [[[3.0, 7.0], [7.0, 3.0]]]]
    np.testing.assert_allclose(output, expected)
    np.testing.assert_allclose(
        basis.apply(output, inverse=True), inputs, rtol=1e-5, atol=1e-5
    )


def test_rank_deficient_matrix_inverse_returns_a_minimum_norm_representative():
    transform = zdx.Map(M=[[1.0, 2.0], [2.0, 4.0]])

    representative = transform.apply([3.0, 6.0], inverse=True)

    np.testing.assert_allclose(representative, [0.6, 1.2], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(transform.apply(representative), [3.0, 6.0], rtol=1e-5)
    reverse = lambda values: transform.apply(values, inverse=True)
    jacobian = jax.jit(jax.jacrev(reverse))(jnp.array([3.0, 6.0]))
    np.testing.assert_allclose(
        jacobian, [[0.04, 0.08], [0.08, 0.16]], rtol=1e-5, atol=1e-5
    )


def test_map_reverse_minimises_the_matrix_stage_before_undoing_nonuniform_scale():
    transform = zdx.Map(s=[2.0, 1.0], M=[[1.0], [1.0]])

    representative = transform.apply([2.0], inverse=True)

    # The matrix stage splits 2 equally into [1,1], then the scale is undone.
    # This is a representative full input; the scale changes its norm geometry.
    np.testing.assert_allclose(representative, [0.5, 1.0], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(transform.apply(representative), [2.0], rtol=1e-5)


def test_one_dimensional_basis_returns_scalar_outputs_and_minimum_norm_inputs():
    transform = zdx.Map(M=[3.0, 4.0])

    np.testing.assert_allclose(transform.apply([1.0, 2.0]), 11.0)
    representatives = transform.apply([5.0, 10.0], inverse=True)
    np.testing.assert_allclose(
        representatives, [[0.6, 0.8], [1.2, 1.6]], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(transform.apply(representatives), [5.0, 10.0], rtol=1e-5)


def test_zero_matrix_inverse_and_forward_reconstruction_return_zero():
    transform = zdx.Map(M=jnp.zeros((2, 3)))
    target = jnp.array([1.0, 2.0, 3.0])

    @eqx.filter_jit
    def inverse_and_reconstruct(model, values):
        representative = model.apply(values, inverse=True)
        return representative, model.apply(representative)

    np.testing.assert_allclose(transform.apply(target, inverse=True), [0.0, 0.0])
    representative, reconstructed = inverse_and_reconstruct(transform, target)
    np.testing.assert_allclose(representative, [0.0, 0.0])
    np.testing.assert_allclose(reconstructed, [0.0, 0.0, 0.0])
    reverse = lambda values: transform.apply(values, inverse=True)
    reconstruct = lambda values: transform.apply(reverse(values))
    np.testing.assert_allclose(jax.jit(jax.jacrev(reverse))(target), jnp.zeros((2, 3)))
    np.testing.assert_allclose(
        jax.jit(jax.jacrev(reconstruct))(target), jnp.zeros((3, 3))
    )


def test_integer_matrix_inverse_keeps_fractional_output():
    transform = zdx.Map(M=[[2, 0], [0, 4]])

    output = transform.apply([1, 1], inverse=True)

    assert jnp.issubdtype(output.dtype, jnp.floating)
    np.testing.assert_allclose(output, [0.5, 0.25], rtol=1e-6)


def test_complex_matrix_inverse_retains_imaginary_components():
    transform = zdx.Map(M=[[1j, 0], [0, 2]])

    output = transform.apply([1.0, 1.0], inverse=True)

    np.testing.assert_allclose(output, [-1j, 0.5], rtol=1e-5, atol=1e-5)


def test_transform_model_gradients_and_vmap_match_the_affine_formula():
    definition = zdx.Map(x=[1.0, 2.0], s=3.0, b=1.0)
    gradient = jax.grad(lambda model: jnp.sum(model.resolve() ** 2))(definition)

    np.testing.assert_allclose(gradient.x, [24.0, 42.0])
    np.testing.assert_allclose(gradient.s, 36.0)
    np.testing.assert_allclose(gradient.b, 22.0)
    values = jax.vmap(zdx.Map(s=3.0, b=1.0).apply)(jnp.array([1.0, 2.0]))
    np.testing.assert_allclose(values, [4.0, 7.0])


@pytest.mark.parametrize("operands", [{"s": [[1.0], [2.0]]}, {"b": [[1.0], [2.0]]}])
def test_map_scale_and_bias_cannot_expand_the_relevant_shape(operands):
    transform = zdx.Map(**operands)

    with pytest.raises(ValueError):
        transform.apply(jnp.ones(2))
    with pytest.raises(ValueError):
        transform.apply(jnp.ones(2), inverse=True)


@pytest.mark.parametrize("matrix", [1.0, jnp.empty((2, 0))])
def test_map_matrix_requires_nonempty_structural_axes(matrix):
    with pytest.raises(ValueError):
        zdx.Map(M=matrix)


def test_map_contraction_and_reverse_require_the_declared_axis_sizes():
    transform = zdx.Map(M=jnp.ones((2, 3, 4)))

    with pytest.raises(ValueError):
        transform.apply(jnp.ones(3))
    with pytest.raises(ValueError):
        transform.apply(jnp.ones((4, 3)), inverse=True)


def test_elementwise_operation_classes_use_the_transform_contract():
    assert issubclass(zdx.Operation, zdx.Transform)
    for operation in (zdx.Exp, zdx.Log, zdx.Pow):
        assert issubclass(operation, zdx.Operation)


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        (2.0, 4.0),
        ([2.0, 3.0], [4.0, 9.0]),
        ([[1.0, 2.0], [3.0, 4.0]], [[1.0, 4.0], [9.0, 16.0]]),
    ],
)
def test_operations_accept_scalar_vector_and_grid_inputs(inputs, expected):
    output = zdx.Pow(p=2).apply(inputs)

    np.testing.assert_allclose(output, expected)


@pytest.mark.parametrize(
    ("operation", "inverse", "inputs", "expected"),
    [
        (zdx.Exp(base=[2.0, 3.0]), False, 2.0, [4.0, 9.0]),
        (
            zdx.Pow(p=[[1], [2]]),
            False,
            [2.0, 3.0, 4.0],
            [[2.0, 3.0, 4.0], [4.0, 9.0, 16.0]],
        ),
        (
            zdx.Exp(base=[[2.0], [3.0]]),
            False,
            [0.0, 1.0, 2.0],
            [[1.0, 2.0, 4.0], [1.0, 3.0, 9.0]],
        ),
        (
            zdx.Log(base=[[2.0], [4.0]]),
            False,
            [1.0, 4.0, 16.0],
            [[0.0, 2.0, 4.0], [0.0, 1.0, 2.0]],
        ),
        (
            zdx.Exp(base=[[2.0], [4.0]]),
            True,
            [1.0, 4.0, 16.0],
            [[0.0, 2.0, 4.0], [0.0, 1.0, 2.0]],
        ),
        (
            zdx.Log(base=[[2.0], [4.0]]),
            True,
            [0.0, 1.0, 2.0],
            [[1.0, 2.0, 4.0], [1.0, 4.0, 16.0]],
        ),
    ],
)
def test_operation_parameters_follow_jax_broadcasting(
    operation, inverse, inputs, expected
):
    output = operation.apply(inputs, inverse=inverse)

    np.testing.assert_allclose(output, expected, rtol=1e-6)


def test_broadcast_operation_supports_compilation_and_input_gradients():
    operation = zdx.Pow(p=[[1], [2]])
    inputs = jnp.array([2.0, 3.0])
    compiled = eqx.filter_jit(lambda model, values: model.apply(values))

    np.testing.assert_allclose(compiled(operation, inputs), [[2.0, 3.0], [4.0, 9.0]])
    gradient = jax.grad(lambda values: compiled(operation, values).sum())(inputs)
    np.testing.assert_allclose(gradient, [5.0, 7.0])


def test_elementwise_inverse_retains_the_shape_expanded_by_broadcasting():
    transform = zdx.Exp(base=[[2.0], [3.0]])
    output = transform.apply([0.0, 1.0, 2.0])

    recovered = transform.apply(output, inverse=True)

    assert recovered.shape == (2, 3)
    np.testing.assert_allclose(recovered, [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], atol=1e-6)


def test_general_power_has_no_selected_inverse_branch():
    with pytest.raises(NotImplementedError):
        zdx.Pow(p=2).apply([4.0, 9.0], inverse=True)


def test_natural_exponential_logarithm_and_complex_scaling_are_reversible():
    values = jnp.array([-1.0, 0.0, 1.0])
    expected = [1 / np.e, 1, np.e]
    np.testing.assert_allclose(zdx.Exp().apply(values), expected, rtol=1e-6)
    np.testing.assert_allclose(
        zdx.Exp().apply(expected, inverse=True), values, atol=1e-6
    )
    np.testing.assert_allclose(zdx.Log().apply(expected), values, atol=1e-6)
    np.testing.assert_allclose(
        zdx.Log().apply(values, inverse=True), expected, rtol=1e-6
    )

    scale = zdx.Map(s=1 + 1j)
    output = scale.apply(jnp.array([1 + 2j, 3 - 1j]))
    np.testing.assert_allclose(output, [-1 + 3j, 4 + 2j])
    np.testing.assert_allclose(scale.apply(output, inverse=True), [1 + 2j, 3 - 1j])


def test_complex_logarithm_uses_the_principal_branch():
    values = jnp.array([1j, -1 + 0j])

    logarithm = zdx.Log().apply(values)

    np.testing.assert_allclose(logarithm, [0.5j * np.pi, 1j * np.pi], atol=1e-6)
    np.testing.assert_allclose(zdx.Exp().apply(logarithm), values, atol=1e-6)


def test_mask_scatter_gather_and_forward_reconstruction_preserve_batches():
    transform = zdx.Mask(mask=[[True, False], [False, True]])
    compact = jnp.array([[2.0, 5.0], [3.0, 7.0]])

    output = eqx.filter_jit(lambda model, values: model.apply(values))(
        transform, compact
    )

    assert transform.shape == (2, 2)
    assert transform.size == 2
    expected = [[[2.0, 0.0], [0.0, 5.0]], [[3.0, 0.0], [0.0, 7.0]]]
    np.testing.assert_allclose(output, expected)
    np.testing.assert_allclose(transform.apply(output, inverse=True), compact)
    representative = transform.apply(jnp.array([[2.0, 9.0], [8.0, 5.0]]), inverse=True)
    np.testing.assert_allclose(
        transform.apply(representative),
        [[2.0, 0.0], [0.0, 5.0]],
    )
    np.testing.assert_allclose(jax.vmap(transform.apply)(compact), output)


def test_mask_can_change_dynamically_with_the_same_shape_and_selected_count():
    transform = zdx.Mask(mask=[[True, False], [False, True]])
    moved = transform.set(mask=jnp.array([[False, True], [True, False]]))
    compiled = eqx.filter_jit(lambda model, values: model.apply(values))
    compact = jnp.array([2.0, 5.0])

    np.testing.assert_allclose(compiled(transform, compact), [[2.0, 0.0], [0.0, 5.0]])
    np.testing.assert_allclose(compiled(moved, compact), [[0.0, 2.0], [5.0, 0.0]])
    gradient = jax.grad(lambda values: jnp.sum(moved.apply(values) ** 2))(compact)
    np.testing.assert_allclose(gradient, [4.0, 10.0])

    reverse = eqx.filter_jit(lambda model, values: model.apply(values, inverse=True))
    full = jnp.array([[7.0, 2.0], [5.0, 9.0]])
    np.testing.assert_allclose(reverse(transform, full), [7.0, 9.0])
    np.testing.assert_allclose(reverse(moved, full), [2.0, 5.0])
    gradient = jax.grad(lambda values: jnp.sum(reverse(moved, values) ** 2))(full)
    np.testing.assert_allclose(gradient, [[0.0, 4.0], [10.0, 0.0]])


@pytest.mark.parametrize(
    ("mask", "error"),
    [([1, 0], TypeError), (True, ValueError), ([False, False], ValueError)],
)
def test_mask_requires_a_nonempty_boolean_selection(mask, error):
    with pytest.raises(error):
        zdx.Mask(mask=mask)


def test_mask_compact_and_full_shapes_must_match_the_selection_layout():
    transform = zdx.Mask(mask=[[True, False], [False, True]])

    with pytest.raises(ValueError):
        transform.apply(jnp.ones(3))
    with pytest.raises(ValueError):
        transform.apply(jnp.ones(4), inverse=True)
