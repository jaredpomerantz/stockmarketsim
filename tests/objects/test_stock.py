"""Tests for the Stock module."""

import numpy as np
import pytest

from simulator.objects.stock import Stock


@pytest.fixture()
def basic_stock() -> Stock:  # noqa: D103
    return Stock(
        cash=500.0,
        earning_value_of_assets=500.0,
        latest_quarterly_earnings=500.0,
        price_history=np.arange(1825),
        quality_of_leadership=0.5,
    )


def test_stock_get_price_changes_over_time_with_valid_price_history_returns_expected_result(  # noqa: D103
    basic_stock,
) -> None:
    # Act.
    result = basic_stock.get_price_changes_over_time()

    # Assert.
    assert np.all(
        np.isclose(
            result,
            np.array(
                [
                    5.48546352e-04,
                    5.51267916e-03,
                    1.67224080e-02,
                    5.19031142e-02,
                    1.09489051e-01,
                    2.50171350e-01,
                    1.50205761e00,
                    1.82399000e05,
                ]
            ),
        )
    )

def test_stock_update_price_history_with_valid_price_returns_expected_result(basic_stock) -> None:  # noqa: D103
    # Act.
    result = basic_stock.update_price_history(1825.0)

    # Assert.
    assert np.all(np.isclose(result, np.arange(1, 1826)))
