from __future__ import annotations

import pytest
import zodiax


def pytest_configure(config):
    zodiax_deprecations = [
        r"ignore:boolean_filter is deprecated as of v0.5.0.*:DeprecationWarning",
        r"ignore:set_array is deprecated as of v0.5.0.*:DeprecationWarning",
        r"ignore:filter_grad is deprecated as of v0.5.0.*:DeprecationWarning",
        r"ignore:filter_value_and_grad is deprecated as of v0.5.0.*:DeprecationWarning",
        r"ignore:fisher.hessian is deprecated as of v0.5.0.*:DeprecationWarning",
        r"ignore:fisher_matrix is deprecated as of v0.5.0.*:DeprecationWarning",
        r"ignore:covariance_matrix is deprecated as of v0.5.0.*:DeprecationWarning",
        r"ignore:scheduler is deprecated as of v0.5.0.*:DeprecationWarning",
        r"ignore:sgd is deprecated as of v0.5.0.*:DeprecationWarning",
        r"ignore:adam is deprecated as of v0.5.0.*:DeprecationWarning",
        r"ignore:get_optimiser is deprecated as of v0.5.0.*:DeprecationWarning",
    ]
    for warning_filter in zodiax_deprecations:
        config.addinivalue_line("filterwarnings", warning_filter)


class A(zodiax.base.Base):
    param: float
    b: B

    def __init__(self, param, b):
        self.param = param
        self.b = b

    def model(self):
        return self.param**2 + self.b.param**2


class B(zodiax.base.Base):
    param: float

    def __init__(self, param):
        self.param = param


@pytest.fixture
def create_base():
    def _create_base(
        param: float = 1.0,
        b: float = 2.0,
    ) -> zodiax.base.Base:
        return A(param, B(b))

    return _create_base
