"""Definition for the Stock class."""

import numpy as np


ZERO_REPLACE_VALUE = 0.01


class Stock:
    """The class definition for a Stock."""

    def __init__(
        self,
        cash: float,
        earning_value_of_assets: float,
        latest_quarterly_earnings: float,
        price_history: np.ndarray,
        quality_of_leadership: float,
    ) -> None:
        """Initialize the Stock class.

        Args:
            cash: Available cash for the company.
            earning_value_of_assets: The value of the assets owned.
                This determines the company's ability to earn cash.
            latest_quarterly_earnings: The latest quarterly earnings by the company.
            price_history: The price history of the stock.
                Should be an array of size (1825,).
            quality_of_leadership: Determines the company's decision-making ability.
                This influences the company's ability to turn cash investments into
                earning_value_of_assets.

        """
        self.cash = cash
        self.earning_value_of_assets = earning_value_of_assets
        self.latest_quarterly_earnings = latest_quarterly_earnings
        self.price_history = price_history
        self.quality_of_leadership = quality_of_leadership

        self.price = self.price_history[-1]

    def get_price_changes_over_time(self) -> np.ndarray:
        """Get the price changes over time.

        Includes price percent change for 1-day, 5-day, 10-day, 1-month,
        3-month, 6-month, 1-year, 3-year, and 5-year periods from the current
        day.
        """
        indices_to_keep = [-2, -11, -31, -91, -181, -366, -1096, 0]
        prices_of_interest = self.price_history[indices_to_keep]
        prices_of_interest = np.where(
            prices_of_interest == 0, ZERO_REPLACE_VALUE, prices_of_interest
        )
        return (self.price - prices_of_interest) / prices_of_interest

    def update_price_history(self, price: float) -> np.ndarray:
        """Update the price history of the stock.

        Args:
            price: The latest price to add.

        Returns:
            An array of the updated price history.

        """
        return np.append(self.price_history[1:], price)
