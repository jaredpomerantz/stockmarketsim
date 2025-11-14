"""Definition for the Participant class."""

import uuid

import numpy as np

from simulator.objects.orders import BuyOrder, SellOrder
from simulator.objects.policies.base_policy import BasePolicy
from simulator.objects.stock import Portfolio


class Participant:
    """The class definition for the Participant.

    Participants own stock and perform trades on the market.
    """

    def __init__(
        self, stock_portfolio: Portfolio, policy: BasePolicy, cash: float
    ) -> None:
        """Initialize the Participant class.

        Args:
            stock_portfolio: The stocks owned by the participant.
            policy: The policy that the participant will use.
            cash: The cash held by the participant.
        """
        self.id = str(uuid.uuid4())
        self.stock_portfolio = stock_portfolio
        self.policy = policy
        self.stock_portfolio_value_history = (
            np.ones(shape=(1825,), dtype=float) * self.stock_portfolio.get_stock_portfolio_value()
        )
        self.cash = cash

        self.policy.initialize_portfolio(self.stock_portfolio)

    def __repr__(self) -> str:
        """Defines the string representation of the Participant."""
        return f"Participant {self.id}"

    def initialize_policy_portfolio(self, stock_portfolio: Portfolio) -> None:
        """Adds the portfolio to the policy.

        Args:
            stock_portfolio: The stocks owned by the participant.
        """
        self.policy.initialize_portfolio(portfolio=stock_portfolio)

    def step_participant(
        self,
    ) -> tuple[list[tuple[BuyOrder, str]], list[tuple[SellOrder, str]]]:
        """Steps the participant forward.

        Returns:
            A tuple of BuyOrders and SellOrders.
        """
        self.stock_portfolio.update_portfolio_value_history()
        self.update_policy()
        buy_orders, sell_orders = self.get_actions()
        buy_orders_with_id = [(buy_order, self.id) for buy_order in buy_orders]
        sell_orders_with_id = [(sell_order, self.id) for sell_order in sell_orders]
        return buy_orders_with_id, sell_orders_with_id

    def get_stock_portfolio_value(self) -> float:
        """Get the total value of the Participant's stock portfolio."""
        return self.stock_portfolio.get_stock_portfolio_value()

    def get_actions(self) -> tuple[list[BuyOrder], list[SellOrder]]:
        """Gets actions from the policy.

        Returns:
            A tuple of BuyOrders and SellOrders.

        """
        buy_orders = self.policy.infer_buy_actions()
        sell_orders = self.policy.infer_sell_actions()
        return (buy_orders, sell_orders)

    def update_policy(self) -> None:
        """Updates the participant's policy based on performance vs market.

        Args:
            market_baseline: The total market's performance.

        """
        self.policy.update_policy()
