"""Definition for the Policy class."""

from abc import abstractmethod

from simulator.objects.orders import BuyOrder, SellOrder
from simulator.objects.stock import Stock


class BasePolicy:
    """The class definition for a Policy.

    Policies in this implementation will have a common framework. Inputs
    will be a defined set of features for different stocks. Outputs will be
    regressions, rounded down to the nearest integer, for each input stock.
    """

    def create_buy_order(self, stock: Stock, stock_quantity: int) -> list[BuyOrder]:
        """Create a buy order for a Stock.

        Args:
            stock: The stock to buy.
            stock_quantity: The number of the stock to buy.

        """
        raise NotImplementedError()

    def create_sell_order(self, stock: Stock, stock_quantity: int) -> list[SellOrder]:
        """Create a sell order for a Stock.

        Args:
            stock: The stock to buy.
            stock_quantity: The number of the stock to buy.

        """
        raise NotImplementedError()

    @abstractmethod
    def infer_buy_actions(self, current_market) -> list[BuyOrder]:
        """Infer buy actions based on the current market.

        Args:
            current_market: The current market of stocks.

        """
        raise NotImplementedError()

    @abstractmethod
    def infer_sell_actions(self, portfolio) -> list[SellOrder]:
        """Infer sell actions based on the current portfolio.

        Args:
            portfolio: The participant's stock portfolio.

        """
        raise NotImplementedError()

    @abstractmethod
    def update(self, policy_performance: float, market_baseline: float) -> None:
        """Updates the policy based on performance relative to market.

        Args:
            policy_performance: The performance of this policy.
            market_baseline: The total market's performance.

        """
        raise NotImplementedError()
