"""Definition for the Market class."""

import numpy as np

from simulator.objects.stock import Stock

MINIMUM_STOCK_PRICE = 1.0
DAYS_IN_A_QUARTER = 92


class Market:
    """The class definition for the Market.

    This class is responsible for facilitating the macro-level market conditions,
    contains all the stocks in the market, moves the market forward over time, and
    creates events that will effect stocks.
    """

    def __init__(self, stocks: list[Stock], interest_rate_apy: float) -> None:
        """Initialize the Market class.

        Args:
            stocks: The stocks to include in the market.
            interest_rate_apy: The interest rate with which stocks' cash grows.
                This is an annualized percentage yield (APY).

        """
        self.stocks = stocks
        self.interest_rate_apy = interest_rate_apy

        self.minimum_stock_price = MINIMUM_STOCK_PRICE

    def advance_stocks(self) -> None:
        """Advances the market's stocks forward one day."""
        stocks_to_delist = []
        for stock in self.stocks:
            stock.price = self.inject_noise_into_stock_price(stock)
            stock.update_price_history(stock.price)
            stock.compound_cash(self.interest_rate_apy)
            if self.check_stock_for_delisting(stock):
                stocks_to_delist.append(stock)

        for stock in stocks_to_delist:
            self.delist_stock(stock)

    def inject_noise_into_stock_price(self, stock: Stock) -> float:
        """Injects random noise into stocks.

        Samples from a 0-mean normal distribution with standard deviation equal to the
        stock's volatility times its price.

        Args:
            stock: The stock object to add noise to.

        Returns:
            An updated stock price.

        """
        updated_price = stock.price + np.random.normal(
            0, stock.stock_volatility * stock.price
        )
        return updated_price

    def check_stock_for_delisting(self, stock: Stock) -> bool:
        """Check if a stock meets the criteria to be delisted.

        A stock will be delisted if any of the following are true:
            Its price dips below the minimum stock price.
            Its quarterly earnings are smaller than the debt that the stock
                accrues in a quarter due to interest. This represents that the company
                that the stock represents can no longer service their debt and are
                going bankrupt.

        Inputs:
            stock: The stock to check for delisting.

        Returns:
            A boolean representing whether the stock is to be delisted.

        """
        if stock.price < self.minimum_stock_price:
            return True
        if stock.cash < 0 and stock.latest_quarterly_earnings < -(
            stock.cash
            * (self.interest_rate_apy + 1) ** (DAYS_IN_A_QUARTER / 365)
        ):
            return True
        return False

    def delist_stock(self, stock: Stock) -> None:
        """Delists a stock.

        This occurs when a stock dips below the minimum stock price or a stock
        goes bankrupt. In this initial implementation, the stock price will be manually
        set to 0, and the stock will be removed from the market.

        Inputs:
            stock: The stock to delist.
        """
        stock.price = 0.0
        self.stocks.remove(stock)
