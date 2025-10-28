"""Definition for the Transformer Policy class."""

import numpy as np
import torch

from simulator.objects.market import Market
from simulator.objects.orders import BuyOrder, SellOrder
from simulator.objects.policies.architectures import ModelTask
from simulator.objects.policies.architectures.base_nn import BaseNN
from simulator.objects.policies.architectures.perceptron import MultiLayerPerceptron
from simulator.objects.policies.base_policy import BasePolicy
from simulator.objects.stock import Portfolio, Stock, StockHolding


class NNPolicy(BasePolicy):
    """The NNPolicy class definition.

    This policy decides which stocks to buy and sell by using a neural network.
    """

    def __init__(
        self,
        market: Market,
        portfolio: Portfolio,
        n_stocks_to_sample: int,
        max_stocks_per_timestep: int,
        buy_selection_model: BaseNN,
        sell_selection_model: BaseNN,
        valuation_model: BaseNN,
    ) -> None:
        """Initializes the NNPolicy.

        Args:
            market: The stock market object to sample from.
            portfolio: The portfolio of the participant.
            n_stocks_to_sample: The number of stocks made visible to the model.
                This is the length of the input tensor that will be given to
                the model to select. For consistency, each input tensor will
                be the same size regardless of the number of stocks in the market.
            max_stocks_per_timestep: The maximum number of buy/sell actions per step.
            buy_selection_model: The model responsible for deciding stocks to buy.
            sell_selection_model: The model responsible for deciding stocks to sell.
            valuation_model: The model responsible for deciding stock value.

        """
        self.market = market
        self.portfolio = portfolio
        self.n_stocks_to_sample = n_stocks_to_sample
        self.max_stocks_per_timestep = max_stocks_per_timestep
        self.valuation_model = valuation_model
        # What if instead, we assign the policy a model that predicts each stock's price in 30 days?
        # For buy, pick the biggest one and give a price (like maybe 10% toward that projected price from the current price).
        # For sell, pick the most bearish one (same with the 10% toward the projected lower price).

    def generate_buy_input_tensor(
        self, market: Market, portfolio: Portfolio, n_stocks: int
    ) -> tuple[list[Stock], torch.Tensor]:
        """Generates an input tensor of randomly selected stocks.

        Args:
            market: The stock market to sample from.
            portfolio: The stocks in the participant's portfolio.
            n_stocks: The number of stocks to sample from the stock market.

        Returns:
            A tensor of stock parameters to input into the model.

        """
        cash_stock = Stock(10.0, 0, 0, np.array([0.0]), 0.5, 0.0)
        cash_holding = StockHolding(cash_stock, 0)

        n_stocks_in_market = len(market.stocks)
        selected_stocks = market.stocks

        # If the number of available stocks is more than the length of the
        # input tensor, sample n_stocks stocks.
        if len(market.stocks) > n_stocks:
            selected_stock_indices = np.random.choice(
                range(n_stocks_in_market), size=n_stocks, replace=False
            )
            selected_stocks = [market.stocks[i] for i in selected_stock_indices]

        valid_stocks = torch.Tensor(
            [
                np.append(
                    stock.get_stock_features(),
                    (
                        portfolio.stock_holding_dict.get(stock.id) or cash_holding
                    ).stock_quantity,
                )
                for stock in selected_stocks
            ]
        )
        return selected_stocks, torch.cat(
            (self.get_cash_features(), valid_stocks), dim=0
        )

    def generate_sell_input_tensor(
        self, market: Market, portfolio: Portfolio, n_stocks: int
    ) -> tuple[list[Stock], torch.Tensor]:
        """Generates an input tensor of randomly selected stocks.

        Args:
            market: The stock market to sample from.
            portfolio: The stocks in the participant's portfolio.
            n_stocks: The number of stocks to sample from the stock market.

        Returns:
            A tensor of stock parameters to input into the model.

        """
        default_stock = Stock(10.0, 0, 0, np.array([0.0]), 0.5, 0.0)
        default_holding = StockHolding(default_stock, 0)

        portfolio_stocks = [
            stock_holding.stock for stock_holding in portfolio.get_stock_holding_list()
        ]
        selected_stocks = [default_stock] + portfolio_stocks
        # If the number of available stocks is more than the length of the
        # input tensor, sample n_stocks stocks.
        if len(list(portfolio.stock_holding_dict.keys())) > n_stocks:
            n_stocks_in_portfolio = len(portfolio_stocks)
            selected_stock_indices = np.random.choice(
                range(n_stocks_in_portfolio), size=n_stocks, replace=False
            )
            selected_stocks = [default_stock] + [
                portfolio_stocks[i] for i in selected_stock_indices
            ]
        valid_stocks = torch.Tensor(
            [
                np.append(
                    stock.get_stock_features(),
                    (
                        portfolio.stock_holding_dict.get(stock.id) or default_holding
                    ).stock_quantity,
                )
                for stock in selected_stocks
            ]
        )
        return selected_stocks, torch.cat(
            (self.get_cash_features(), valid_stocks), dim=0
        )

    def infer_buy_actions(self) -> list[BuyOrder]:
        """Infer buy actions based on the current market."""
        best_stock_index = 1
        buy_orders = []
        portfolio_after_buys = Portfolio(self.portfolio.get_stock_holding_list())
        while len(buy_orders) < self.max_stocks_per_timestep:
            selected_stocks, input_tensor = self.generate_buy_input_tensor(
                self.market, portfolio_after_buys, self.n_stocks_to_sample
            )
            valuations = self.valuation_model.forward(input_tensor)
            current_prices = torch.tensor([stock.price for stock in selected_stocks])
            projected_percent_change = (valuations - current_prices) / current_prices

            best_stock_index = int(torch.argmax(projected_percent_change).item())
            if best_stock_index == 0:
                return buy_orders

            current_stock_price = selected_stocks[best_stock_index].price
            stock_valuation = float(valuations[best_stock_index].item())
            price_to_offer = (
                current_stock_price + (stock_valuation - current_stock_price) * 0.1
            )
            buy_orders.append(
                BuyOrder(
                    stock=selected_stocks[best_stock_index],
                    quantity=1,
                    price=price_to_offer,
                )
            )
            portfolio_after_buys.add_stock_holding(
                StockHolding(selected_stocks[best_stock_index], 1)
            )
        return buy_orders

    def infer_sell_actions(self) -> list[SellOrder]:
        """Infer sell actions based on the current portfolio."""
        best_stock_index = 1
        sell_orders = []
        portfolio_after_sells = Portfolio(self.portfolio.get_stock_holding_list())
        while len(sell_orders) < self.max_stocks_per_timestep:
            selected_stocks, input_tensor = self.generate_sell_input_tensor(
                self.market, portfolio_after_sells, self.n_stocks_to_sample
            )
            valuations = self.valuation_model.forward(input_tensor)
            current_prices = torch.tensor([stock.price for stock in selected_stocks])
            projected_percent_change = (valuations - current_prices) / current_prices

            worst_stock_index = int(torch.argmin(projected_percent_change).item())
            if worst_stock_index == 0:
                return sell_orders

            current_stock_price = selected_stocks[worst_stock_index].price
            stock_valuation = float(valuations[best_stock_index].item())
            price_to_offer = (
                current_stock_price + (stock_valuation - current_stock_price) * 0.1
            )

            sell_orders.append(
                SellOrder(
                    stock=selected_stocks[worst_stock_index],
                    quantity=1,
                    price=price_to_offer,
                )
            )
            portfolio_after_sells.add_stock_holding(
                StockHolding(selected_stocks[worst_stock_index], 1)
            )
        return sell_orders

    def update(self, policy_performance: float, market_baseline: float) -> None:
        """Updates the policy based on performance relative to market.

        Args:
            policy_performance: The performance of this policy.
            market_baseline: The total market's performance.

        """
        loss = market_baseline - policy_performance

    def get_cash_features(self) -> torch.Tensor:
        """Gets the features for the cash asset."""
        time_periods = np.array([1825, 1095, 365, 180, 90, 30, 10, 5, 1])
        compounded_rates = torch.tensor(
            (self.market.interest_rate_apy + 1) ** (time_periods / 365)
        )
        null_cash_array = torch.tensor([0.0])
        return torch.cat(
            (torch.tensor([10.0, 10.0, 10.0, 10.0]), compounded_rates, null_cash_array),
            dim=0,
        ).unsqueeze(0)
