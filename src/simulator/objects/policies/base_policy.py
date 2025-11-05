"""Definition for the Policy class."""

from abc import abstractmethod

from simulator.objects.orders import BuyOrder, SellOrder
from simulator.objects.stock import Stock, Portfolio
from simulator.objects.market import Market


class BasePolicy:
    """The class definition for a Policy.

    Policies in this implementation will have a common framework. Inputs
    will be a defined set of features for different stocks. Outputs will be
    regressions, rounded down to the nearest integer, for each input stock.
    """

    def __init__(self, market: Market, portfolio: Portfolio) -> None:
        """Initializes the BasePolicy.

        Args:
            market: The stock market object to sample from.
            portfolio: The portfolio of the participant.

        """
        self.market = market
        self.portfolio = portfolio

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
    def infer_buy_actions(self) -> list[BuyOrder]:
        """Infer buy actions based on the current market."""
        raise NotImplementedError()

    @abstractmethod
    def infer_sell_actions(self) -> list[SellOrder]:
        """Infer sell actions based on the current portfolio."""
        raise NotImplementedError()

    @abstractmethod
    def update_policy(self) -> None:
        """Updates the policy based on performance relative to market."""
        raise NotImplementedError()
