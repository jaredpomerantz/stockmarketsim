"""Definition for the Policy class."""

from abc import abstractmethod

import numpy as np
import torch

from simulator.objects.orders import BuyOrder, SellOrder
from simulator.objects.stock import Portfolio, Stock, StockHolding


class BasePolicy:
    """The class definition for a Policy.

    Policies in this implementation will have a common framework. Inputs
    will be a defined set of features for different stocks. Outputs will be
    regressions, rounded down to the nearest integer, for each input stock.
    """

    def __init__(
        self, market, n_stocks_to_sample: int, max_stocks_per_timestep: int
    ) -> None:
        """Initializes the BasePolicy.

        Args:
            market: The stock market object to sample from.
            n_stocks_to_sample: The number of stocks made visible to the model.
                This is the length of the input tensor that will be given to
                the model to select. For consistency, each input tensor will
                be the same size regardless of the number of stocks in the market.
            max_stocks_per_timestep: The maximum number of buy/sell actions per step.
        """
        self.market = market
        self.n_stocks_to_sample = n_stocks_to_sample
        self.max_stocks_per_timestep = max_stocks_per_timestep
        self.cash_stock = Stock(1.0, 0, 0, np.array([1.0]), 0.5, 0.0)

    def initialize_portfolio(self, portfolio: Portfolio) -> None:
        """Initializes the portfolio for the policy.

        Args:
            portfolio: The portfolio of the participant.
        """
        self.portfolio = portfolio

    def generate_generic_buy_input_array(self) -> tuple[list[Stock], np.ndarray]:
        """Generates an input tensor of randomly selected stocks.

        Args:
            market: The stock market to sample from.
            portfolio: The stocks in the participant's portfolio.
            n_stocks: The number of stocks to sample from the stock market.

        Returns:
            A tensor of stock parameters to input into the model.
        """
        cash_holding = StockHolding(self.cash_stock, 0)

        n_stocks_in_market = len(self.market.stocks)
        selected_stocks = self.market.stocks

        # If the number of available stocks is more than the length of the
        # input tensor, sample n_stocks stocks.
        if len(self.market.stocks) > self.n_stocks_to_sample:
            selected_stock_indices = np.random.choice(
                range(n_stocks_in_market), size=self.n_stocks_to_sample, replace=False
            )
            selected_stocks = [self.market.stocks[i] for i in selected_stock_indices]

        return selected_stocks, np.array(
            [
                np.append(
                    stock.get_stock_features(),
                    (
                        self.portfolio.stock_holding_dict.get(stock.id) or cash_holding
                    ).stock_quantity,
                )
                for stock in selected_stocks
            ]
        )

    def generate_generic_sell_input_array(self) -> tuple[list[Stock], np.ndarray]:
        """Generates an input tensor of randomly selected stocks.

        Args:
            market: The stock market to sample from.
            portfolio: The stocks in the participant's portfolio.
            n_stocks: The number of stocks to sample from the stock market.

        Returns:
            A tensor of stock parameters to input into the model.
        """
        cash_holding = StockHolding(self.cash_stock, 0)

        portfolio_stocks = [
            stock_holding.stock
            for stock_holding in self.portfolio.get_stock_holding_list()
        ]
        selected_stocks = portfolio_stocks

        # If the number of available stocks is more than the length of the
        # input tensor, sample n_stocks stocks.
        if (
            len(list(self.portfolio.stock_holding_dict.keys()))
            > self.n_stocks_to_sample
        ):
            n_stocks_in_portfolio = len(portfolio_stocks)
            selected_stock_indices = np.random.choice(
                range(n_stocks_in_portfolio),
                size=self.n_stocks_to_sample,
                replace=False,
            )
            selected_stocks = [portfolio_stocks[i] for i in selected_stock_indices]

        valid_stock_feature_array = np.array(
            [
                np.append(
                    stock.get_stock_features(),
                    (
                        self.portfolio.stock_holding_dict.get(stock.id) or cash_holding
                    ).stock_quantity,
                )
                for stock in selected_stocks
            ]
        )
        return selected_stocks, valid_stock_feature_array

    def get_buy_actions_from_valuations(
        self, valuations: np.ndarray, selected_stocks: list[Stock]
    ) -> list[BuyOrder]:
        """Produces a list of BuyActions from stock valuations.

        Args:
            valuations: An array of valuations for the associated selected_stocks.
            selected_stocks: The stocks that the policy will choose from.

        Returns:
            A list of BuyOrders that the policy will output.
        """
        buy_orders = []
        current_prices = np.array([stock.price for stock in selected_stocks])
        projected_percent_change = (valuations - current_prices) / (
            current_prices + 1e-8
        )
        projected_percent_change[0] = np.clip(projected_percent_change[0], 0.98, 1.05)

        value_stock_pairs: list[tuple[float, float, float, Stock]] = [
            (
                projected_percent_change[i].item(),
                valuations[i].item(),
                current_prices[i].item(),
                selected_stocks[i],
            )
            for i in range(len(selected_stocks))
        ]

        sorted_value_stock_pairs = sorted(value_stock_pairs, reverse=True)

        for i in range(self.max_stocks_per_timestep):
            _, valuation, current_price, selected_stock = sorted_value_stock_pairs[i]

            if selected_stock is self.cash_stock:
                return buy_orders

            price_to_offer = current_price + (valuation - current_price) * 0.1
            buy_orders.append(
                BuyOrder(
                    stock=selected_stock,
                    quantity=1,
                    price=price_to_offer,
                )
            )
        return buy_orders

    def get_sell_actions_from_valuations(
        self, valuations: np.ndarray, selected_stocks: list[Stock]
    ) -> list[SellOrder]:
        """Produces a list of SellOrders from stock valuations.

        Args:
            valuations: An array of valuations for the associated selected_stocks.
            selected_stocks: The stocks that the policy will choose from.

        Returns:
            A list of SellOrders that the policy will output.
        """
        sell_orders = []
        current_prices = np.array([stock.price for stock in selected_stocks])
        projected_percent_change = (valuations - current_prices) / current_prices
        projected_percent_change[0] = np.clip(projected_percent_change[0], 0.98, 1.05)

        value_stock_pairs: list[tuple[float, float, float, Stock]] = [
            (
                projected_percent_change[i].item(),
                valuations[i].item(),
                current_prices[i].item(),
                selected_stocks[i],
            )
            for i in range(len(selected_stocks))
        ]

        sorted_value_stock_pairs = sorted(value_stock_pairs)

        for i in range(self.max_stocks_per_timestep):
            _, valuation, current_price, selected_stock = sorted_value_stock_pairs[i]

            if selected_stock is self.cash_stock:
                return sell_orders

            price_to_offer = current_price + (valuation - current_price) * 0.1
            sell_orders.append(
                SellOrder(
                    stock=selected_stock,
                    quantity=1,
                    price=price_to_offer,
                )
            )
        return sell_orders

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
