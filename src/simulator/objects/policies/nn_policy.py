"""Definition for the Transformer Policy class."""

from pathlib import Path

import numpy as np
import torch

from simulator.objects.market import Market
from simulator.objects.orders import BuyOrder, SellOrder
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
        valuation_model_path: Path,
        valuation_model_noise_std: float = 0.0,
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
            valuation_model_path: The path to model responsible for valuing stocks.
            valuation_model_noise_std: The standard deviation of noise for weights.

        """
        super().__init__(market=market, portfolio=portfolio)

        self.n_stocks_to_sample = n_stocks_to_sample
        self.max_stocks_per_timestep = max_stocks_per_timestep

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.valuation_model = torch.load(valuation_model_path, weights_only=False).to(
            self.device
        )
        self.valuation_model_noise_std = valuation_model_noise_std
        self.inject_noise_into_model_weights()

        self.cash_stock = Stock(10.0, 0, 0, np.array([0.0]), 0.5, 0.0)

        self.selected_stock_history = []
        self.buy_valuation_history = []
        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(params=self.valuation_model.parameters())

    def inject_noise_into_model_weights(self) -> None:
        """Injects random noise into valuation model weights."""
        with torch.no_grad():
            for param in self.valuation_model.parameters():
                noise = torch.randn(param.size()) * self.valuation_model_noise_std
                param.add_(noise.to(self.device))

    def generate_buy_input_tensor(self) -> tuple[list[Stock], torch.Tensor]:
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
        valid_stocks = torch.Tensor(
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
        return [self.cash_stock] + selected_stocks, torch.cat(
            (self.get_cash_features(), valid_stocks), dim=0
        ).to(torch.float32)

    def generate_sell_input_tensor(self) -> tuple[list[Stock], torch.Tensor]:
        """Generates an input tensor of randomly selected stocks.

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

        valid_stocks = torch.Tensor(
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
        return [self.cash_stock] + selected_stocks, torch.cat(
            (self.get_cash_features(), valid_stocks), dim=0
        ).to(torch.float32)

    def infer_buy_actions(self) -> list[BuyOrder]:
        """Infer buy actions based on the current market."""
        best_stock_index = 1
        buy_orders = []
        selected_stocks, input_tensor = self.generate_buy_input_tensor()
        valuations = self.valuation_model.forward(input_tensor.to(self.device))

        self.update_valuation_histories(
            selected_stocks=selected_stocks, valuations=valuations
        )

        current_prices = torch.tensor([stock.price for stock in selected_stocks]).to(
            self.device
        )
        projected_percent_change = (valuations - current_prices) / (
            current_prices + 1e-8
        )

        value_stock_pairs = [
            (
                projected_percent_change[i],
                valuations[i],
                current_prices[i],
                selected_stocks[i],
            )
            for i in range(len(selected_stocks))
        ]

        sorted_value_stock_pairs = sorted(value_stock_pairs)

        for i in range(self.max_stocks_per_timestep):
            _, valuation, current_price, selected_stock = sorted_value_stock_pairs[i]

            if selected_stock is self.cash_stock:
                return buy_orders

            price_to_offer = (current_price + (valuation - current_price) * 0.1).item()
            buy_orders.append(
                BuyOrder(
                    stock=selected_stocks[best_stock_index],
                    quantity=1,
                    price=price_to_offer,
                )
            )
        return buy_orders

    def infer_sell_actions(self) -> list[SellOrder]:
        """Infer sell actions based on the current portfolio."""
        best_stock_index = 1
        sell_orders = []
        selected_stocks, input_tensor = self.generate_sell_input_tensor()
        valuations = self.valuation_model.forward(input_tensor.to(self.device))
        current_prices = torch.tensor([stock.price for stock in selected_stocks]).to(
            self.device
        )
        projected_percent_change = (valuations - current_prices) / current_prices

        value_stock_pairs = [
            (
                projected_percent_change[i],
                valuations[i],
                current_prices[i],
                selected_stocks[i],
            )
            for i in range(len(selected_stocks))
        ]

        sorted_value_stock_pairs = sorted(value_stock_pairs)

        for i in range(self.max_stocks_per_timestep):
            _, valuation, current_price, selected_stock = sorted_value_stock_pairs[i]

            if selected_stock is self.cash_stock:
                return sell_orders

            price_to_offer = (current_price + (valuation - current_price) * 0.1).item()
            sell_orders.append(
                SellOrder(
                    stock=selected_stocks[best_stock_index],
                    quantity=1,
                    price=price_to_offer,
                )
            )
        return sell_orders

    def update_policy(self) -> None:
        """Updates the policy based on performance relative to market."""
        if len(self.selected_stock_history) < 30:
            return
        past_stocks = self.selected_stock_history[0]
        past_stocks_valuations = self.buy_valuation_history[0].to(self.device)

        past_stocks_current_prices = (
            torch.tensor([stock.price or 0.0 for stock in past_stocks])
            .to(self.device)
            .to(torch.float32)
        )

        loss_val = self.loss(past_stocks_valuations, past_stocks_current_prices)

        self.optimizer.zero_grad()
        loss_val.backward()
        self.optimizer.step()

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

    def update_valuation_histories(
        self, selected_stocks: list[Stock], valuations: torch.Tensor
    ) -> None:
        """Updates the valuation histories with their respective stocks.

        Args:
            selected_stocks: A list of stocks that were selected to value.
            valuations: The valuations of the respective selected_stocks]

        """
        if len(self.selected_stock_history) >= 30:
            self.selected_stock_history = self.selected_stock_history[1:]
            self.buy_valuation_history = self.buy_valuation_history[1:]

        self.selected_stock_history.append(selected_stocks)
        self.buy_valuation_history.append(valuations)
