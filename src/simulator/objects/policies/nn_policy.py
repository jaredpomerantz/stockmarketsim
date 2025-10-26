"""Definition for the Transformer Policy class."""

import numpy as np
import torch

from simulator.objects.market import Market
from simulator.objects.orders import BuyOrder, SellOrder
from simulator.objects.policies.base_policy import BasePolicy
from simulator.objects.stock import Portfolio, Stock, StockHolding
from simulator.objects.policies.architectures.perceptron import MultiLayerPerceptron
from simulator.objects.policies.architectures import ModelTask
from simulator.objects.policies.architectures.base_nn import BaseNN


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

        """
        self.market = market
        self.portfolio = portfolio
        self.n_stocks_to_sample = n_stocks_to_sample
        self.max_stocks_per_timestep = max_stocks_per_timestep
        self.buy_selection_model = buy_selection_model
        self.sell_selection_model = sell_selection_model
        self.valuation_model = valuation_model

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
        stop_entry = torch.tensor([[-1.0] * 12])
        default_stock = Stock(0, 0, 0, np.array([0.0]), 0.5, 0.0)
        default_holding = StockHolding(default_stock, 0)

        # If the number of available stocks is less than the length of the
        # input tensor, pad the tensor.
        if len(market.stocks) <= n_stocks:
            valid_stocks = torch.tensor(
                [
                    np.append(
                        stock.get_stock_features(),
                        (
                            portfolio.stock_holding_dict.get(stock.id, None)
                            or default_holding
                        ).stock_quantity,
                    ).astype(float)
                    for stock in market.stocks
                ]
            )
            padding_stocks = torch.cat(
                [stop_entry for _ in range(n_stocks - valid_stocks.shape[0])], dim=0
            )
            selected_stocks = market.stocks + [
                default_stock for _ in range(n_stocks - len(market.stocks))
            ]
            return selected_stocks, torch.cat(
                (stop_entry, valid_stocks, padding_stocks), dim=0
            )

        # If the number of available stocks is more than the length of the
        # input tensor, sample n_stocks stocks.
        n_stocks_in_market = len(market.stocks)
        selected_stock_indices = np.random.choice(
            range(n_stocks_in_market), size=n_stocks, replace=False
        )
        selected_stocks = [market.stocks[i] for i in selected_stock_indices]
        return selected_stocks, torch.Tensor(
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
        stop_entry = torch.tensor([[-1.0] * 12])
        default_stock = Stock(0, 0, 0, np.array([0.0]), 0.5, 0.0)
        default_holding = StockHolding(default_stock, 0)
        portfolio_stocks = [stock_holding.stock for stock_holding in portfolio.get_stock_holding_list()]

        # If the number of stocks in the portfolio is less than the length of the
        # input tensor, pad the tensor.
        if len(list(portfolio.stock_holding_dict.keys())) <= n_stocks:
            valid_stocks = torch.tensor(
                [
                    np.append(
                        stock.get_stock_features(),
                        (
                            portfolio.stock_holding_dict.get(stock.id, None)
                            or default_holding
                        ).stock_quantity,
                    ).astype(float)
                    for stock in portfolio_stocks
                ]
            )
            padding_stocks = torch.cat(
                [stop_entry for _ in range(n_stocks - valid_stocks.shape[0])], dim=0
            )
            selected_stocks = portfolio_stocks + [
                default_stock for _ in range(n_stocks - len(portfolio_stocks))
            ]
            return selected_stocks, torch.cat(
                (stop_entry, valid_stocks, padding_stocks), dim=0
            )

        n_stocks_in_portfolio = len(portfolio_stocks)
        selected_stock_indices = np.random.choice(
            range(n_stocks_in_portfolio), size=n_stocks, replace=False
        )
        selected_stocks = [portfolio_stocks[i] for i in selected_stock_indices]
        return selected_stocks, torch.Tensor(
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

    def infer_buy_actions(self) -> list[BuyOrder]:
        """Infer buy actions based on the current market."""
        last_selection = 1
        buy_orders = []
        portfolio_after_buys = Portfolio(
            self.portfolio.get_stock_holding_list()
        )
        while len(buy_orders) < self.max_stocks_per_timestep and last_selection != 0:
            selected_stocks, input_tensor = self.generate_buy_input_tensor(
                self.market, portfolio_after_buys, self.n_stocks_to_sample
            )
            stock_index = self.buy_selection_model.forward(input_tensor)
            if input_tensor[stock_index] == torch.Tensor([-1] * 12):
                return buy_orders

            stock_valuation = self.valuation_model.forward(input_tensor[stock_index])

            buy_orders.append(
                BuyOrder(
                    stock=selected_stocks[stock_index],
                    quantity=1,
                    price=stock_valuation.item(),
                )
            )
            portfolio_after_buys.add_stock_holding(StockHolding(selected_stocks[stock_index], 1))
        return buy_orders

    def infer_sell_actions(self, portfolio) -> list[SellOrder]:
        """Infer sell actions based on the current portfolio.

        Args:
            portfolio: The participant's stock portfolio.

        """
        last_selection = 1
        sell_orders = []
        portfolio_after_sells = Portfolio(
            self.portfolio.get_stock_holding_list()
        )
        while len(sell_orders) < self.max_stocks_per_timestep and last_selection != 0:
            selected_stocks, input_tensor = self.generate_sell_input_tensor(
                self.market, portfolio_after_sells, self.n_stocks_to_sample
            )
            stock_index = self.sell_selection_model.forward(input_tensor)
            if input_tensor[stock_index] == torch.Tensor([-1] * 12):
                return sell_orders

            stock_valuation = self.valuation_model.forward(input_tensor[stock_index])

            sell_orders.append(
                BuyOrder(
                    stock=selected_stocks[stock_index],
                    quantity=1,
                    price=stock_valuation.item(),
                )
            )
            portfolio_after_sells.add_stock_holding(StockHolding(selected_stocks[stock_index], 1))
        return sell_orders
