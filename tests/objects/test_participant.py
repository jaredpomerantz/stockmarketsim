"""Tests for the Participant module."""

import pytest

from simulator.objects.orders import BuyOrder, SellOrder
from simulator.objects.participant import Participant
from simulator.objects.stock import Portfolio, StockHolding

from .test_market import basic_market  # noqa: F401
from .test_policies import example_policy  # noqa: F401


@pytest.fixture()
def example_participant(basic_market, example_policy) -> Participant:  # noqa: F811
    example_portfolio = Portfolio([StockHolding(basic_market.stocks[0], 1)])
    return Participant(
        stock_portfolio=example_portfolio, policy=example_policy, cash=1000
    )


def test_participant_get_actions_returns_valid_actions(example_participant) -> None:
    # Act.
    buy_actions, sell_actions = example_participant.get_actions()

    # Assert.
    if buy_actions:
        assert isinstance(buy_actions[0], BuyOrder)

    if sell_actions:
        assert isinstance(sell_actions[0], SellOrder)


def test_participant_update_policy_completes(example_participant) -> None:
    # Arrange.
    [example_participant.get_actions() for _ in range(30)]

    # Act.
    example_participant.update_policy()

    # Assert.
    assert len(example_participant.policy.selected_stock_history) == 30
