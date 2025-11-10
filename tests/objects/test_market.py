"""Tests for the Market module."""

import numpy as np
import pytest

from simulator.objects.market import Market
from simulator.objects.orders import BuyOrder, SellOrder
from simulator.objects.participant import Participant
from simulator.objects.policies.nn_policy import NNPolicy
from simulator.objects.stock import Portfolio, Stock, StockHolding

from . import TEST_MODULE


@pytest.fixture()
def stock1() -> Stock:
    return Stock(
        cash=-10000000.0,
        earning_value_of_assets=500.0,
        latest_quarterly_earnings=500.0,
        price_history=np.arange(1825),
        quality_of_leadership=0.5,
        stock_volatility=0.01,
    )


@pytest.fixture()
def stock2() -> Stock:
    return Stock(
        cash=0.5,
        earning_value_of_assets=500.0,
        latest_quarterly_earnings=500.0,
        price_history=np.append(np.ones(1824, dtype=float), 0.5),
        quality_of_leadership=0.5,
        stock_volatility=0.01,
    )


@pytest.fixture()
def stock3() -> Stock:
    return Stock(
        cash=500.0,
        earning_value_of_assets=500.0,
        latest_quarterly_earnings=500.0,
        price_history=np.arange(1825),
        quality_of_leadership=0.5,
        stock_volatility=0.01,
    )


@pytest.fixture()
def market_with_no_participants(stock1, stock2, stock3) -> Market:
    return Market(stocks=[stock1, stock2, stock3], interest_rate_apy=0.03)


@pytest.fixture()
def participant1(stock1, market_with_no_participants) -> Participant:
    return Participant(
        stock_portfolio=Portfolio([StockHolding(stock=stock1, stock_quantity=3)]),
        policy=NNPolicy(
            market=market_with_no_participants,
            n_stocks_to_sample=3,
            max_stocks_per_timestep=3,
            valuation_model_path=TEST_MODULE / "models" / "model.pt",
            valuation_model_noise_std=0.001,
        ),
        cash=1000,
    )


@pytest.fixture()
def participant2(stock2, stock3, market_with_no_participants) -> Participant:
    return Participant(
        stock_portfolio=Portfolio(
            [
                StockHolding(stock=stock2, stock_quantity=1),
                StockHolding(stock=stock3, stock_quantity=1),
            ]
        ),
        policy=NNPolicy(
            market=market_with_no_participants,
            n_stocks_to_sample=3,
            max_stocks_per_timestep=3,
            valuation_model_path=TEST_MODULE / "models" / "model.pt",
            valuation_model_noise_std=0.001,
        ),
        cash=1000,
    )


@pytest.fixture()
def basic_market(market_with_no_participants, participant1, participant2) -> Market:
    market_with_no_participants.add_participants([participant1, participant2])
    return market_with_no_participants


def test_market_advance_stocks_has_expected_result(basic_market) -> None:
    # Act.
    basic_market.advance_stocks()

    # Assert.
    assert len(basic_market.stocks) == 1


def test_market_resolve_market_orders_executes_single_trade_successfully(
    stock1, stock2, stock3, basic_market, participant1, participant2
) -> None:
    # Arrange.
    expected_participant1_portfolio = Portfolio([StockHolding(stock1, 2)])
    expected_participant1_cash = 1047.5
    expected_participant2_portfolio = Portfolio(
        [StockHolding(stock1, 1), StockHolding(stock2, 1), StockHolding(stock3, 1)]
    )
    expected_participant2_cash = 952.5

    # Act.
    basic_market.resolve_market_orders(
        buy_orders=[(BuyOrder(stock1, 1, 50.0), participant2.id)],
        sell_orders=[(SellOrder(stock1, 1, 45.0), participant1.id)],
    )

    # Assert.
    assert participant1.stock_portfolio == expected_participant1_portfolio
    assert participant1.cash == expected_participant1_cash
    assert participant2.stock_portfolio == expected_participant2_portfolio
    assert participant2.cash == expected_participant2_cash


def test_market_resolve_market_orders_executes_multiple_trades_successfully(
    stock1, stock2, stock3, basic_market, participant1, participant2
) -> None:
    # Arrange.
    expected_participant1_portfolio = Portfolio([StockHolding(stock1, 2), StockHolding(stock2, 1)])
    expected_participant1_cash = 992.5
    expected_participant2_portfolio = Portfolio(
        [StockHolding(stock1, 1), StockHolding(stock3, 1)]
    )
    expected_participant2_cash = 1007.5

    # Act.
    basic_market.resolve_market_orders(
        buy_orders=[
            (BuyOrder(stock1, 1, 50.0), participant2.id),
            (BuyOrder(stock2, 1, 60.0), participant1.id),
        ],
        sell_orders=[
            (SellOrder(stock1, 1, 45.0), participant1.id),
            (SellOrder(stock2, 1, 50.0), participant2.id),
        ],
    )

    # Assert.
    assert participant1.stock_portfolio == expected_participant1_portfolio
    assert participant1.cash == expected_participant1_cash
    assert participant2.stock_portfolio == expected_participant2_portfolio
    assert participant2.cash == expected_participant2_cash


def test_market_resolve_market_orders_with_low_buyer_cash_does_not_execute_single_trade(
    stock1, stock2, stock3, basic_market, participant1, participant2
) -> None:
    # Arrange.
    participant2.cash = 0
    expected_participant1_portfolio = participant1.stock_portfolio
    expected_participant1_cash = participant1.cash
    expected_participant2_portfolio = participant2.stock_portfolio
    expected_participant2_cash = participant2.cash

    # Act.
    basic_market.resolve_market_orders(
        buy_orders=[(BuyOrder(stock1, 1, 50.0), participant2.id)],
        sell_orders=[(SellOrder(stock1, 1, 45.0), participant1.id)],
    )

    # Assert.
    assert participant1.stock_portfolio == expected_participant1_portfolio
    assert participant1.cash == expected_participant1_cash
    assert participant2.stock_portfolio == expected_participant2_portfolio
    assert participant2.cash == expected_participant2_cash
