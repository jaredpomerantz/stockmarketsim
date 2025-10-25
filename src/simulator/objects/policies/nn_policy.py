"""Definition for the Transformer Policy class."""

import numpy as np
import torch

from simulator.objects.market import Market
from simulator.objects.policies.base_policy import BasePolicy
from simulator.objects.stock import Portfolio, Stock, StockHolding


class NNPolicy(BasePolicy):
    """The NNPolicy class definition.

    This policy decides which stocks to buy and sell by using a neural network.
    """

    def __init__(
        self, market: Market, portfolio: Portfolio, n_stocks_to_sample: int
    ) -> None:
        """Initializes the NNPolicy.

        Args:
            market: The stock market object to sample from.
            portfolio: The portfolio of the participant.
            n_stocks_to_sample: The number of stocks made visible to the model.
                This is the length of the input tensor that will be given to
                the model to select. For consistency, each

        """
        self.market = market
        self.portfolio = portfolio
        self.n_stocks_to_sample = n_stocks_to_sample
        # include parameters for a neural network.

    def generate_input_tensor(
        self, market: Market, portfolio: Portfolio, n_stocks: int
    ) -> torch.Tensor:
        """Generates an input tensor of randomly selected stocks.

        Args:
            market: The stock market to sample from.
            portfolio: The stocks in the participant's portfolio.
            n_stocks: The number of stocks to sample from the stock market.

        Returns:
            A tensor of stock parameters to input into the model.

        """
        stop_entry = torch.tensor([[-1.0] * 12])
        default_holding = StockHolding(Stock(0, 0, 0, np.array([0.0]), 0.5, 0.0), 0)
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
            return torch.cat((stop_entry, valid_stocks, padding_stocks), dim=0)

        n_stocks_in_market = len(market.stocks)
        selected_stock_indices = np.random.choice(
            range(n_stocks_in_market), size=n_stocks, replace=False
        )
        return torch.Tensor(
            [
                np.append(
                    market.stocks[i].get_stock_features(),
                    (
                        portfolio.stock_holding_dict.get(market.stocks[i].id)
                        or default_holding
                    ).stock_quantity,
                )
                for i in selected_stock_indices
            ]
        )
