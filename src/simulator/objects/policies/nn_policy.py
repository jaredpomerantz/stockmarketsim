"""Definition for the Neural Network Policy class."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from simulator.objects.market import Market
from simulator.objects.orders import BuyOrder, SellOrder
from simulator.objects.policies.base_policy import BasePolicy
from simulator.objects.stock import Stock, StockHolding

if TYPE_CHECKING:
    from simulator.objects.market import Market


class NNPolicy(BasePolicy):
    """The NNPolicy class definition.

    This policy decides which stocks to buy and sell by using a neural network.
    """

    def __init__(
        self,
        market: Market,
        n_stocks_to_sample: int,
        max_stocks_per_timestep: int,
        valuation_model_path: Path,
        valuation_model_noise_std: float = 0.0,
    ) -> None:
        """Initializes the NNPolicy.

        Args:
            market: The stock market object to sample from.
            n_stocks_to_sample: The number of stocks made visible to the model.
                This is the length of the input tensor that will be given to
                the model to select. For consistency, each input tensor will
                be the same size regardless of the number of stocks in the market.
            max_stocks_per_timestep: The maximum number of buy/sell actions per step.
            valuation_model_path: The path to model responsible for valuing stocks.
            valuation_model_noise_std: The standard deviation of noise for weights.
        """
        super().__init__(
            market=market,
            n_stocks_to_sample=n_stocks_to_sample,
            max_stocks_per_timestep=max_stocks_per_timestep,
        )

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.valuation_model: torch.nn.Module = torch.load(
            valuation_model_path, weights_only=False
        ).to(self.device)
        self.valuation_model_noise_std = valuation_model_noise_std
        self.inject_noise_into_model_weights()

        self.selected_stock_history = []
        self.buy_input_tensor_history = []
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

        Returns:
            A tensor of stock parameters to input into the model.
        """
        selected_stocks, valid_stocks_feature_array = (
            self.generate_generic_buy_input_array()
        )

        valid_stocks_features = torch.Tensor(valid_stocks_feature_array)
        return [self.cash_stock] + selected_stocks, torch.cat(
            (self.get_cash_features(), valid_stocks_features), dim=0
        ).to(torch.float32)

    def generate_sell_input_tensor(self) -> tuple[list[Stock], torch.Tensor]:
        """Generates an input tensor of randomly selected stocks.

        Returns:
            A tensor of stock parameters to input into the model.
        """
        selected_stocks, valid_stock_feature_array = (
            self.generate_generic_sell_input_array()
        )
        valid_stocks_features = torch.Tensor(valid_stock_feature_array)
        return [self.cash_stock] + selected_stocks, torch.cat(
            (self.get_cash_features(), valid_stocks_features), dim=0
        ).to(torch.float32)

    def infer_buy_actions(self) -> list[BuyOrder]:
        """Infer buy actions based on the current market."""
        selected_stocks, input_tensor = self.generate_buy_input_tensor()

        self.update_buy_input_tensor_histories(
            selected_stocks=selected_stocks, input_tensor=input_tensor
        )
        valuations: np.ndarray = (
            torch.clamp(
                self.valuation_model.forward(input_tensor.to(self.device)),
                min=0.0,
                max=1e8,
            )
            .detach()
            .cpu()
            .numpy()
        )

        return self.get_buy_actions_from_valuations(
            valuations=valuations, selected_stocks=selected_stocks
        )

    def infer_sell_actions(self) -> list[SellOrder]:
        """Infer sell actions based on the current portfolio."""
        selected_stocks, input_tensor = self.generate_sell_input_tensor()
        if len(list(self.portfolio.get_stock_holding_list())) == 0:
            return []

        valuations: np.ndarray = (
            torch.clamp(
                self.valuation_model.forward(input_tensor.to(self.device)),
                min=0.0,
                max=1e8,
            )
            .detach()
            .cpu()
            .numpy()
        )
        return self.get_sell_actions_from_valuations(
            valuations=valuations, selected_stocks=selected_stocks
        )

    def update_policy(self) -> None:
        """Updates the policy based on performance relative to market."""
        if len(self.selected_stock_history) < 30:
            return
        past_stocks: torch.Tensor = self.buy_input_tensor_history[0]

        past_stocks_current_prices = (
            torch.tensor(
                [stock.price or 0.0 for stock in self.selected_stock_history[0]]
            )
            .to(self.device)
            .to(torch.float32)
        )
        past_stock_valuations = self.valuation_model(past_stocks.to(self.device))

        loss_val = self.loss(past_stock_valuations, past_stocks_current_prices)

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

    def update_buy_input_tensor_histories(
        self, selected_stocks: list[Stock], input_tensor: torch.Tensor
    ) -> None:
        """Updates the valuation histories with their respective stocks.

        Args:
            selected_stocks: A list of stocks that were selected to value.
            input_tensor: The input_tensor of the respective selected_stocks.
        """
        if len(self.selected_stock_history) >= 30:
            self.selected_stock_history = self.selected_stock_history[1:]
            self.buy_input_tensor_history = self.buy_input_tensor_history[1:]

        self.selected_stock_history.append(selected_stocks)
        self.buy_input_tensor_history.append(input_tensor)
