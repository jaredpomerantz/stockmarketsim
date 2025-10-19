"""Definition for the Participant class."""

from simulator.objects.stock import StockHolding


class Participant:
    """The class definition for the Participant.

    Participants own stock and perform trades on the market.
    """

    def __init__(self, stock_portfolio: list[StockHolding]) -> None:
        """Initialize the Participant class.

        Args:
            stock_portfolio: The stocks owned by the participant.

        """
        self.stock_portfolio = stock_portfolio

    def get_stock_portfolio_value(self) -> float:
        """Get the total value of the Participant's stock portfolio."""
        return sum(
            [
                stock_holding.get_holding_value()
                for stock_holding in self.stock_portfolio
            ]
        )
