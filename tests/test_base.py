from __future__ import annotations
import equinox as eqx
import jax.numpy as np
import jax.random as jr
import pytest
import zodiax
import zodiax as zdx


from jax import config

config.update("jax_debug_nans", True)


@pytest.mark.parametrize(
    "wrapped",
    [
        ["a", ["b", ["c", ["d"]]]],
        [[[["a"], "b"], "c"], "d"],
        ["a", "b", "c", "d"],
    ],
)
def test_unwrap(wrapped):
    assert zodiax.base._unwrap(wrapped) == ["a", "b", "c", "d"]


@pytest.mark.parametrize(
    "wrapped,values,expected",
    [
        (["a", ["b", "c", "d"]], [1, 2], [1, 2, 2, 2]),
        ([["a", "b", "c"], "d"], [1, 2], [1, 1, 1, 2]),
        (["a", "b", "c", "d"], [1, 2, 3, 4], [1, 2, 3, 4]),
    ],
)
def test_unwrap_with_values(wrapped, values, expected):
    assert zodiax.base._unwrap(wrapped, values)[1] == expected


@pytest.mark.parametrize(
    "paths",
    [
        ["a.b", ["b.c", ["c.d", ["d.e"]]]],
        [[[["a.b"], "b.c"], "c.d"], "d.e"],
        ["a.b", "b.c", "c.d", "d.e"],
    ],
)
def test_format(paths):
    expected = [["a", "b"], ["b", "c"], ["c", "d"], ["d", "e"]]
    assert zodiax.base._format(paths) == expected


@pytest.mark.parametrize(
    "paths,values,expected",
    [
        (["a.b", ["b.c", "c.d", "d.e"]], [1, 2], [1, 2, 2, 2]),
        ([["a.b", "b.c", "c.d"], "d.e"], [1, 2], [1, 1, 1, 2]),
        (["a.b", "b.c", "c.d", "d.e"], [1, 2, 3, 4], [1, 2, 3, 4]),
    ],
)
def test_format_with_values(paths, values, expected):
    assert zodiax.base._format(paths, values)[1] == expected


class TestBase:
    @pytest.mark.parametrize(
        "parameters,expected",
        [
            ("param", 1.0),
            (["param", "b.param"], [1.0, 2.0]),
        ],
    )
    def test_get(self, create_base, parameters, expected):
        assert create_base().get(parameters) == expected

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

    def test_update(self, create_base):
        output = create_base().update({"param": 10.0, "b.param": 5.0})
        assert output.get(["param", "b.param"]) == [10.0, 5.0]


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
