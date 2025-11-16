"""Definition for the LightGBM Policy class."""

import pickle
from pathlib import Path

import numpy as np

from simulator.objects.market import Market
from simulator.objects.orders import BuyOrder, SellOrder
from simulator.objects.policies.base_policy import BasePolicy
from simulator.objects.stock import Stock


class MLPolicy(BasePolicy):
    """Class definition for the MLPolicy."""

    def __init__(
        self,
        market: Market,
        n_stocks_to_sample: int,
        max_stocks_per_timestep: int,
        valuation_model_path: Path,
    ) -> None:
        """Initializes the MLPolicy class."""
        super().__init__(
            market=market,
            n_stocks_to_sample=n_stocks_to_sample,
            max_stocks_per_timestep=max_stocks_per_timestep,
        )

        self.n_stocks_to_sample = n_stocks_to_sample
        self.max_stocks_per_timestep = max_stocks_per_timestep

        with valuation_model_path.open("rb") as model_file:
            self.valuation_model = pickle.load(model_file)

        self.selected_stock_history = []
        self.buy_input_history = []

    def generate_buy_input_array(self) -> tuple[list[Stock], np.ndarray]:
        """Generates an input array of randomly selected stocks.

        Returns:
            An of stock parameters to input into the model.
        """
        selected_stocks, valid_stock_features = self.generate_generic_buy_input_array()
        return [self.cash_stock] + selected_stocks, np.concatenate(
            (self.get_cash_features(), valid_stock_features), axis=0
        )

    def generate_sell_input_array(
        self,
    ) -> tuple[list[Stock], np.ndarray]:
        """Generates an input array of randomly selected stocks.

        Returns:
            An of stock parameters to input into the model.
        """
        selected_stocks, valid_stock_features = self.generate_generic_sell_input_array()
        return [self.cash_stock] + selected_stocks, np.concatenate(
            (self.get_cash_features(), valid_stock_features), axis=0
        )

    def infer_buy_actions(self) -> list[BuyOrder]:
        """Infer buy actions based on the current market."""
        selected_stocks, input_array = self.generate_buy_input_array()

        self.update_buy_input_array_histories(
            selected_stocks=selected_stocks, input_array=input_array
        )
        valuations: np.ndarray = np.clip(
            self.valuation_model.predict(input_array), a_min=0.05, a_max=1e8
        )  # type: ignore
        return self.get_buy_actions_from_valuations(
            valuations=valuations, selected_stocks=selected_stocks
        )

    def infer_sell_actions(self) -> list[SellOrder]:
        """Infer sell actions based on the current portfolio."""
        selected_stocks, input_array = self.generate_sell_input_array()

        valuations: np.ndarray = np.clip(
            self.valuation_model.predict(input_array), a_min=0.05, a_max=1e8
        )  # type: ignore
        return self.get_sell_actions_from_valuations(
            valuations=valuations, selected_stocks=selected_stocks
        )

    def update_policy(self) -> None:
        """Updates the policy based on performance relative to market."""
        if len(self.selected_stock_history) < 30:
            return
        past_stocks: np.ndarray = self.buy_input_history[0]

        past_stocks_current_prices = np.array(
            [stock.price or 0.0 for stock in self.selected_stock_history[0]]
        )
        self.valuation_model.fit(past_stocks, past_stocks_current_prices)

    def get_cash_features(self) -> np.ndarray:
        """Gets the features for the cash asset."""
        time_periods = np.array([1825, 1095, 365, 180, 90, 30, 10, 5, 1])
        compounded_rates = np.array(
            (self.market.interest_rate_apy + 1) ** (time_periods / 365)
        )
        null_cash_array = np.array([0.0])
        return np.expand_dims(
            np.concatenate(
                (
                    np.array([10.0, 10.0, 10.0, 10.0]),
                    compounded_rates,
                    null_cash_array,
                ),
                axis=0,
            ),
            axis=0,
        )

    def update_buy_input_array_histories(
        self, selected_stocks: list[Stock], input_array: np.ndarray
    ) -> None:
        """Updates the valuation histories with their respective stocks.

        Args:
            selected_stocks: A list of stocks that were selected to value.
            input_array: The input_tensor of the respective selected_stocks.
        """
        if len(self.selected_stock_history) >= 30:
            self.selected_stock_history = self.selected_stock_history[1:]
            self.buy_input_history = self.buy_input_history[1:]

        self.selected_stock_history.append(selected_stocks)
        self.buy_input_history.append(input_array)


class RandomForestPolicy(MLPolicy):
    """Class definition for the RandomForestPolicy."""

    def __init__(
        self,
        market: Market,
        n_stocks_to_sample: int,
        max_stocks_per_timestep: int,
        valuation_model_path: Path,
    ) -> None:
        """Initializes the RandomForestPolicy class."""
        super().__init__(
            market=market,
            n_stocks_to_sample=n_stocks_to_sample,
            max_stocks_per_timestep=max_stocks_per_timestep,
            valuation_model_path=valuation_model_path,
        )

    def update_policy(self) -> None:
        """A policy update function.

        Partial fitting is not available for RandomForests, so policies cannot be
        updated incrementally.
        """
        return


class PassiveAggressivePolicy(MLPolicy):
    """Class definition for the PassiveAggressivePolicy."""

    def __init__(
        self,
        market: Market,
        n_stocks_to_sample: int,
        max_stocks_per_timestep: int,
        valuation_model_path: Path,
    ) -> None:
        """Initializes the PassiveAggressivePolicy class."""
        super().__init__(
            market=market,
            n_stocks_to_sample=n_stocks_to_sample,
            max_stocks_per_timestep=max_stocks_per_timestep,
            valuation_model_path=valuation_model_path,
        )

    def update_policy(self) -> None:
        """A policy update function for the PassiveAggressiveRegressor."""
        if len(self.selected_stock_history) < 30:
            return
        past_stocks: np.ndarray = self.buy_input_history[0]

        past_stocks_current_prices = np.array(
            [stock.price or 0.0 for stock in self.selected_stock_history[0]]
        )
        self.valuation_model.partial_fit(past_stocks, past_stocks_current_prices)
