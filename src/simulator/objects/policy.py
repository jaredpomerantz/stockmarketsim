"""Definition for the Policy class."""


from abc import abstractmethod

from simulator.objects.stock import Stock


class Policy:
    """The class definition for a Policy.

    Policies in this implementation will have a common framework. Inputs
    will be a defined set of features for different stocks. Outputs will be
    regressions, rounded down to the nearest integer, for each input stock.
    """

    def create_buy_order(self, stock_quantity: int, stock: Stock) -> None:
        """Create a buy order for a Stock."""
        raise NotImplementedError

    def create_sell_order(self, stock_quantity: int, stock: Stock) -> None:
        """Create a sell order for a Stock."""
        raise NotImplementedError
    
    @abstractmethod
    def infer_buy_actions(self, )
