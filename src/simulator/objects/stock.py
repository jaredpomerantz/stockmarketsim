"""Definition for the Stock class."""

import uuid

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
        stock_volatility: float,
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
            stock_volatility: Determines the day-to-day volatility of the stock.
                This volatility represents a single standard deviation in a normal
                distribution for a day's percent change due to randomness for the stock.

        """
        self.id = str(uuid.uuid4())
        self.cash = cash
        self.earning_value_of_assets = earning_value_of_assets
        self.latest_quarterly_earnings = latest_quarterly_earnings
        self.price_history = price_history
        self.quality_of_leadership = quality_of_leadership
        self.stock_volatility = stock_volatility

        self.price = self.price_history[-1]

    def __repr__(self) -> str:
        return f"(Stock {self.id}; Price {self.price}; Cash {self.cash})"

    def get_price_changes_over_time(self) -> np.ndarray:
        """Get the price changes over time.

        Includes price percent change for 1-day, 5-day, 10-day, 1-month,
        3-month, 6-month, 1-year, 3-year, and 5-year periods from the current
        day.
        """
        indices_to_keep = np.array([-2, -5, -11, -31, -91, -181, -366, -1096, 0])
        prices_of_interest = self.price_history[indices_to_keep]
        prices_of_interest = np.where(
            prices_of_interest == 0, ZERO_REPLACE_VALUE, prices_of_interest
        )
        return (self.price - prices_of_interest) / prices_of_interest

    def get_stock_features(self) -> np.ndarray:
        """Gets the stock features as a numpy array.

        Returns:
            An array of stock features.

        """
        output = np.array(
            [
                self.price,
                self.cash,
                self.earning_value_of_assets,
                self.latest_quarterly_earnings,
            ]
        )
        return np.append(output, self.get_price_changes_over_time()).astype(float)

    def update_price_history(self, price: float) -> np.ndarray:
        """Update the price history of the stock.

        Args:
            price: The latest price to add.

        Returns:
            An array of the updated price history.

        """
        return np.append(self.price_history[1:], price)

    def compound_cash(self, interest_rate_apy: float) -> None:
        """Compounds the current cash holding.

        This initial implementation assumes that the stock can go into debt freely.
        Cash is compounded the same whether it's negative or positive.
        """
        self.cash *= (interest_rate_apy + 1) ** (1 / 365)


class StockHolding:
    """The class definition of a holding of a Stock."""

    def __init__(self, stock: Stock, stock_quantity: int) -> None:
        """Instantiate the StockHolding class."""
        self.stock = stock
        self.stock_quantity = stock_quantity

    def get_holding_value(self) -> float:
        """Calculate the value of the StockHolding.

        Returns:
            The value of the holding.

        """
        return self.stock.price * self.stock_quantity


class Portfolio:
    """The class definition of a portfolio of StockHoldings."""

    def __init__(self, stock_holdings: list[StockHolding]) -> None:
        """Instantiate the Portfolio class."""
        self.stock_holding_dict = self.get_stock_holding_dictionary(stock_holdings)

    def get_stock_holding_dictionary(
        self, stock_holdings: list[StockHolding]
    ) -> dict[str, StockHolding]:
        """Gets the stock holding dictionary.

        This dictionary allows StockHoldings to be accessed by stock ID.

        Args:
            stock_holdings: The stock holdings to create a dictionary from.

        Returns:
            A dictionary mapping stock ids to stock holdings.

        """
        self.stock_holding_dict = {}
        for stock_holding in stock_holdings:
            self.stock_holding_dict = self.add_stock_holding(stock_holding)
        return self.stock_holding_dict

    def add_stock_holding(self, stock_holding: StockHolding) -> dict[str, StockHolding]:
        """Adds a stock holding to the portfolio.

        Args:
            stock_holding: The stock_holding to add to the portfolio.

        Returns:
            An updated stock holding dictionary.

        """
        if stock_holding.stock.id in self.stock_holding_dict:
            self.stock_holding_dict[
                stock_holding.stock.id
            ].stock_quantity += stock_holding.stock_quantity
        else:
            self.stock_holding_dict[stock_holding.stock.id] = stock_holding

        return self.stock_holding_dict

    def remove_stock_holding(
        self, stock_holding: StockHolding
    ) -> dict[str, StockHolding]:
        """Adds a stock holding to the portfolio.

        Args:
            stock_holding: The stock_holding to add to the portfolio.

        Returns:
            An updated stock holding dictionary.
        """
        if stock_holding.stock.id not in self.stock_holding_dict:
            return self.stock_holding_dict
        if (
            stock_holding.stock_quantity
            >= self.stock_holding_dict[stock_holding.stock.id].stock_quantity
        ):
            del self.stock_holding_dict[stock_holding.stock.id]
        else:
            self.stock_holding_dict[
                stock_holding.stock.id
            ].stock_quantity -= stock_holding.stock_quantity
        return self.stock_holding_dict

    def get_stock_holding_list(self) -> list[StockHolding]:
        """Gets a stock holding list from a stock holding dictionary.

        Returns:
            A list of stock holdings.

        """
        return list(self.stock_holding_dict.values())

    def get_stock_portfolio_value(self) -> float:
        """Gets the total stock portfolio value.

        Returns:
            The value of the stock portfolio.
        """
        return sum(
            stock_holding.stock.price * stock_holding.stock_quantity
            for stock_holding in self.get_stock_holding_list()
        )
