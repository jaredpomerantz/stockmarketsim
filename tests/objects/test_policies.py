"""Tests for the Policies module."""

from pathlib import Path

import pytest
import torch

from simulator.objects.market import Market
from simulator.objects.orders import BuyOrder, SellOrder
from simulator.objects.policies.nn_policy import NNPolicy
from simulator.objects.stock import Portfolio, StockHolding

from .test_market import basic_market  # noqa: F401

TEST_MODULE = Path(__file__).parent


@pytest.fixture()
def example_policy(basic_market: Market) -> NNPolicy:  # noqa: F811
    example_portfolio = Portfolio([StockHolding(basic_market.stocks[0], 1)])
    return NNPolicy(
        market=basic_market,
        portfolio=example_portfolio,
        n_stocks_to_sample=5,
        max_stocks_per_timestep=10,
        valuation_model_path=TEST_MODULE / "models" / "model.pt",
        valuation_model_noise_std=0.01,
    )


def test_nn_policy_generate_buy_input_tensor_given_valid_input_returns_expected_result(
    example_policy,
) -> None:
    # Arrange.
    expected_result = torch.Tensor(
        [
            [
                1.0000e01,
                1.0000e01,
                1.0000e01,
                1.0000e01,
                1.1593e00,
                1.0927e00,
                1.0300e00,
                1.0147e00,
                1.0073e00,
                1.0024e00,
                1.0008e00,
                1.0004e00,
                1.0001e00,
                0.0000e00,
            ],
            [
                1824.0,
                -10000000.0,
                500.0,
                500.0,
                5.4855e-04,
                2.1978e-03,
                5.5127e-03,
                1.6722e-02,
                5.1903e-02,
                1.0949e-01,
                2.5017e-01,
                1.5021e00,
                1.8240e05,
                1.0,
            ],
            [
                0.5,
                0.5,
                500.0,
                500.0,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                0.0,
            ],
            [
                1824.0,
                500.0,
                500.0,
                500.0,
                5.4855e-04,
                2.1978e-03,
                5.5127e-03,
                1.6722e-02,
                5.1903e-02,
                1.0949e-01,
                2.5017e-01,
                1.5021e00,
                1.8240e05,
                0.0,
            ],
        ]
    ).to(torch.float32)

    # Act.
    _, output = example_policy.generate_buy_input_tensor()

    # Assert.
    assert torch.allclose(output, expected_result, atol=1e-4)


def test_nn_policy_infer_buy_actions_completes(example_policy) -> None:
    # Act.
    buy_actions = example_policy.infer_buy_actions()

    # Assert.
    if buy_actions:
        assert isinstance(buy_actions[0], BuyOrder)
    else:
        assert buy_actions == []


def test_nn_policy_generate_sell_input_tensor_given_valid_input_returns_expected_result(
    example_policy,
) -> None:
    # Arrange.
    expected_result = torch.Tensor(
        [
            [
                1.0000e01,
                1.0000e01,
                1.0000e01,
                1.0000e01,
                1.1593e00,
                1.0927e00,
                1.0300e00,
                1.0147e00,
                1.0073e00,
                1.0024e00,
                1.0008e00,
                1.0004e00,
                1.0001e00,
                0.0000e00,
            ],
            [
                1824.0,
                -10000000.0,
                500.0,
                500.0,
                5.4855e-04,
                2.1978e-03,
                5.5127e-03,
                1.6722e-02,
                5.1903e-02,
                1.0949e-01,
                2.5017e-01,
                1.5021e00,
                1.8240e05,
                1.0,
            ],
        ]
    ).to(torch.float32)

    # Act.
    _, output = example_policy.generate_sell_input_tensor()

    # Assert.
    assert torch.allclose(output, expected_result, atol=1e-4)


def test_nn_policy_infer_sell_actions_completes(example_policy) -> None:
    # Act.
    sell_actions = example_policy.infer_sell_actions()

    # Assert.
    if sell_actions:
        assert isinstance(sell_actions[0], SellOrder)
    else:
        assert sell_actions == []
