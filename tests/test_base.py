from __future__ import annotations

import equinox as eqx
import jax.numpy as np
import jax.random as jr
import pytest
import zodiax as zdx

from jax import config

config.update("jax_debug_nans", True)


class TestBase:
    def test_get_leaf_dict_list_and_missing_key(self):
        pytree = {"a": [{"b": 3.0}]}
        assert zdx.base._get_leaf(pytree, ["a", "0", "b"]) == 3.0

        with pytest.raises(KeyError):
            zdx.base._get_leaf(1.0, ["a"])

    def test_unwrap_raises_on_value_length_mismatch(self):
        with pytest.raises(ValueError):
            zdx.base._unwrap(["param", "b.param"], [1.0, 2.0, 3.0])

    def test_unwrap_parameters_only_nested(self):
        output = zdx.base._unwrap(["param", ["b.param", ("c.param",)]])
        assert output == ["param", "b.param", "c.param"]

    def test_normalise_mutation_inputs_error_paths(self):
        with pytest.raises(TypeError):
            zdx.base._normalise_mutation_inputs(
                parameters={"param": 1.0},
                values=2.0,
                method_name="set",
            )

        with pytest.raises(TypeError):
            zdx.base._normalise_mutation_inputs(method_name="set")

        with pytest.raises(TypeError):
            zdx.base._normalise_mutation_inputs(
                parameters="param",
                values=None,
                method_name="add",
                require_values=True,
            )

    @pytest.mark.parametrize(
        "parameters,expected",
        [
            ("param", 1.0),
            (["param", "b.param"], [1.0, 2.0]),
            (("param", "b.param"), [1.0, 2.0]),
            (["param", ("b.param",)], [1.0, 2.0]),
        ],
    )
    def test_get(self, create_base, parameters, expected):
        assert create_base().get(parameters) == expected

    def test_get_as_dict_single_and_multi(self, create_base):
        base = create_base()
        assert base.get("param", as_dict=True) == {"param": 1.0}
        assert base.get(["param", "b.param"], as_dict=True) == {
            "param": 1.0,
            "b.param": 2.0,
        }

    def test_get_to_array_casts_output(self, create_base):
        base = create_base(param=1, b=2)
        output = base.get("param", to_array=True)
        assert hasattr(output, "dtype")
        assert np.issubdtype(output.dtype, np.floating)

    @pytest.mark.parametrize(
        "method,parameters,values,expected",
        [
            ("set", "param", 10.0, 10.0),
            ("add", "param", 10.0, 11.0),
            ("multiply", "param", 10.0, 10.0),
            ("divide", "param", 10.0, 0.1),
            ("power", "param", 3.0, 1.0),
            ("min", "param", 0.5, 0.5),
            ("max", "param", 10.0, 10.0),
        ],
    )
    def test_single_path_ops(self, create_base, method, parameters, values, expected):
        output = getattr(create_base(), method)(parameters, values)
        assert np.allclose(output.get(parameters), expected)

    @pytest.mark.parametrize(
        "method", ["set", "add", "multiply", "divide", "power", "min", "max"]
    )
    def test_multi_path_ops(self, create_base, method):
        output = getattr(create_base(), method)(["param", "b.param"], [2.0, 3.0])
        assert len(output.get(["param", "b.param"])) == 2

    @pytest.mark.parametrize(
        "parameters,values",
        [
            (["param", "b.param"], [10.0, 5.0]),
            (("param", "b.param"), (10.0, 5.0)),
            (["param", ("b.param",)], [10.0, (5.0,)]),
            (("param", ["b.param"]), (10.0, [5.0])),
        ],
    )
    def test_set_path_container_parity(self, create_base, parameters, values):
        output = create_base().set(parameters, values)
        assert output.get(["param", "b.param"]) == [10.0, 5.0]

    def test_set_accepts_mapping(self, create_base):
        output = create_base().set({"param": 10.0, "b.param": 5.0})
        assert output.get(["param", "b.param"]) == [10.0, 5.0]

    def test_set_accepts_tuple_key(self, create_base):
        output = create_base().set({("param", "b.param"): 6.0})
        assert output.get(["param", "b.param"]) == [6.0, 6.0]

    def test_set_accepts_mixed_tuple_and_string_keys(self, create_base):
        output = create_base().set({("param",): 6.0, "b.param": 7.0})
        assert output.get(["param", "b.param"]) == [6.0, 7.0]

    def test_set_accepts_kwargs(self, create_base):
        output = create_base().set(param=6.0, **{"b.param": 7.0})
        assert output.get(["param", "b.param"]) == [6.0, 7.0]

    @pytest.mark.parametrize(
        "parameters",
        [
            ["param", "b.param"],
            ("param", "b.param"),
        ],
    )
    def test_set_accepts_none_positional(self, create_base, parameters):
        output = create_base().set(parameters, None)
        assert output.get(["param", "b.param"]) == [None, None]

    def test_set_accepts_none_single_string_path(self, create_base):
        output = create_base().set("param", None)
        assert output.get("param") is None

    def test_set_accepts_none_mapping(self, create_base):
        output = create_base().set({"param": None, "b.param": None})
        assert output.get(["param", "b.param"]) == [None, None]

    def test_set_accepts_none_kwargs(self, create_base):
        output = create_base().set(param=None, **{"b.param": None})
        assert output.get(["param", "b.param"]) == [None, None]

    def test_set_accepts_partial_none_positional(self, create_base):
        output = create_base().set(["param", "b.param"], [None, 7.0])
        assert output.get(["param", "b.param"]) == [None, 7.0]

    def test_set_accepts_partial_none_mapping(self, create_base):
        output = create_base().set({"param": None, "b.param": 7.0})
        assert output.get(["param", "b.param"]) == [None, 7.0]

    def test_set_accepts_partial_none_kwargs(self, create_base):
        output = create_base().set(param=None, **{"b.param": 7.0})
        assert output.get(["param", "b.param"]) == [None, 7.0]

    @pytest.mark.parametrize(
        "method,mapping,expected",
        [
            ("set", {"param": 10.0, "b.param": 5.0}, [10.0, 5.0]),
            ("add", {"param": 1.0, "b.param": 2.0}, [2.0, 4.0]),
            ("multiply", {"param": 10.0, "b.param": 3.0}, [10.0, 6.0]),
            ("divide", {"param": 2.0, "b.param": 4.0}, [0.5, 0.5]),
            ("power", {"param": 3.0, "b.param": 2.0}, [1.0, 4.0]),
            ("min", {"param": 0.5, "b.param": 3.0}, [0.5, 2.0]),
            ("max", {"param": 10.0, "b.param": 1.0}, [10.0, 2.0]),
        ],
    )
    def test_mutators_accept_mapping(self, create_base, method, mapping, expected):
        output = getattr(create_base(), method)(mapping)
        assert output.get(["param", "b.param"]) == expected

    @pytest.mark.parametrize(
        "method,kwargs,expected",
        [
            ("set", {"param": 10.0, "b.param": 5.0}, [10.0, 5.0]),
            ("add", {"param": 1.0, "b.param": 2.0}, [2.0, 4.0]),
            ("multiply", {"param": 10.0, "b.param": 3.0}, [10.0, 6.0]),
            ("divide", {"param": 2.0, "b.param": 4.0}, [0.5, 0.5]),
            ("power", {"param": 3.0, "b.param": 2.0}, [1.0, 4.0]),
            ("min", {"param": 0.5, "b.param": 3.0}, [0.5, 2.0]),
            ("max", {"param": 10.0, "b.param": 1.0}, [10.0, 2.0]),
        ],
    )
    def test_mutators_accept_kwargs(self, create_base, method, kwargs, expected):
        output = getattr(create_base(), method)(**kwargs)
        assert output.get(["param", "b.param"]) == expected

    def test_mutators_reject_mixed_styles(self, create_base):
        with pytest.raises(TypeError):
            create_base().set(["param"], [10.0], param=1.0)
        with pytest.raises(TypeError):
            create_base().add({"param": 1.0}, param=2.0)


class Foo(zdx.WrapperHolder):

    def __init__(self, nn):
        values, structure = zdx.build_wrapper(nn)
        self.values = values
        self.structure = structure

    def __call__(self, x):
        return self.build(x)


def test_WrapperHolder():
    eqx_model = eqx.nn.MLP(
        in_size=16, out_size=16, width_size=32, depth=1, key=jr.PRNGKey(0)
    )

    foo = Foo(eqx_model)
    x = np.ones(16)

    assert isinstance(foo, zdx.WrapperHolder)
    assert np.allclose(foo(x), eqx_model(x))
    assert np.allclose(foo.multiply("values", 0.0)(x), np.zeros_like(x))


def test_WrapperHolder_missing_attr_raises():
    eqx_model = eqx.nn.MLP(
        in_size=16, out_size=16, width_size=32, depth=1, key=jr.PRNGKey(1)
    )
    foo = Foo(eqx_model)
    with pytest.raises(AttributeError):
        _ = foo.not_a_real_attribute


def test_WrapperHolder_getattr_delegates_to_structure():
    eqx_model = eqx.nn.MLP(
        in_size=16, out_size=16, width_size=32, depth=1, key=jr.PRNGKey(2)
    )
    foo = Foo(eqx_model)
    assert foo.shapes == foo.structure.shapes
