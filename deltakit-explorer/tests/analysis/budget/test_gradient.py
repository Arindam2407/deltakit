from collections.abc import Callable
import itertools
from deltakit_explorer.analysis.budget.gradient.schemes import (
    FirstOrderDerivativeCentralDifference,
)
import pytest
import numpy


class TestCentralDifference:
    @pytest.mark.skip(reason="Gradient computation done through scipy at the moment.")
    @pytest.mark.parametrize(
        "order,c", itertools.product(range(1, 10), (0.1, 0.5, 2, 10))
    )
    def test_generic_case(self, order: int, c: float):
        scheme = FirstOrderDerivativeCentralDifference(order, c)
        n = (order + 1) // 2
        numpy.testing.assert_allclose(
            scheme.required_multiples_of_h,
            [
                numpy.sign(i) / c ** (numpy.abs(i) - 1.0)
                for i in range(-n, n + 1)
                if i != 0
            ],
        )
        assert len(scheme._coefficients) == len(scheme._required_multiples_of_h)

    @pytest.mark.skip(reason="Gradient computation done through scipy at the moment.")
    @pytest.mark.parametrize("c", (0.1, 0.5, 2, 10))
    def test_second_order_scheme(self, c: float):
        scheme = FirstOrderDerivativeCentralDifference(2, c)
        numpy.testing.assert_allclose(scheme.required_multiples_of_h, [-1, 1])

    @pytest.mark.skip(reason="Gradient computation done through scipy at the moment.")
    @pytest.mark.parametrize("order", range(1, 10))
    def test_raise_on_c_too_close_to_1(self, order: int):
        match_str = "^Cannot have a scaling factor too close to 1.$"
        with pytest.raises(RuntimeError, match=match_str):
            FirstOrderDerivativeCentralDifference(order, 1)

    @pytest.mark.skip(reason="Gradient computation done through scipy at the moment.")
    @pytest.mark.parametrize(
        "order,c,func_and_derivative",
        itertools.product(
            range(1, 10),
            (0.1, 0.5, 0.9, 1.1, 2, 10),
            [
                (numpy.exp, numpy.exp),
                (lambda x: 1 / x, lambda x: -1 / x**2),
                (numpy.log, lambda x: 1 / x),
                (numpy.sin, numpy.cos),
                (numpy.arctan, lambda x: 1 / (1 + x**2))
            ],
        ),
    )
    def test_on_analytic_functions(
        self, order: int, c: float, func_and_derivative: tuple[Callable[[float], float],Callable[[float], float]]
    ):
        func, derivative = func_and_derivative
        scheme = FirstOrderDerivativeCentralDifference(order, c)
        h = 1e-5
        x = 1.0

        values = [func(x + dh * h) for dh in scheme.required_multiples_of_h]
        estimation = scheme.approximate(values, h)
        assert pytest.approx(estimation) == derivative(x)
