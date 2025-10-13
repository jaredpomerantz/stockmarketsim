"""Tests for the Market module."""

import numpy as np
import pytest

from simulator.objects.market import Market
from simulator.objects.stock import Stock


@pytest.fixture()
def basic_market() -> Market:
    stock1 = Stock(
        cash=-10000000.0,
        earning_value_of_assets=500.0,
        latest_quarterly_earnings=500.0,
        price_history=np.arange(1825),
        quality_of_leadership=0.5,
        stock_volatility=0.01,
    )
    stock2 = Stock(
        cash=0.5,
        earning_value_of_assets=500.0,
        latest_quarterly_earnings=500.0,
        price_history=np.append(np.ones(1824, dtype=float), 0.5),
        quality_of_leadership=0.5,
        stock_volatility=0.01,
    )
    stock3 = Stock(
        cash=500.0,
        earning_value_of_assets=500.0,
        latest_quarterly_earnings=500.0,
        price_history=np.arange(1825),
        quality_of_leadership=0.5,
        stock_volatility=0.01,
    )

    return Market(stocks=[stock1, stock2, stock3], interest_rate_apy=0.03)


def test_market_advance_stocks_has_expected_result(basic_market) -> None:
    # Act.
    basic_market.advance_stocks()

    # Assert.
    assert len(basic_market.stocks) == 1
