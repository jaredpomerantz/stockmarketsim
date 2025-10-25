"""Tests for the Policies module."""

import pytest
import torch

from simulator.objects.market import Market
from simulator.objects.policies.nn_policy import NNPolicy
from simulator.objects.stock import Portfolio, StockHolding


@pytest.fixture()
def example_policy(basic_market: Market) -> NNPolicy:
    example_portfolio = Portfolio([StockHolding(basic_market.stocks[0], 1)])
    return NNPolicy(basic_market, example_portfolio, 5, 10)


def test_nn_policy_generate_input_tensor_given_valid_input_returns_expected_result(
    example_policy,
) -> None:
    # Arrange.
    expected_result = torch.Tensor(
        [
            [-1] * 12,
            [
                1824.0,
                -10000000.0,
                500.0,
                500.0,
                5.4855e-04,
                5.5127e-03,
                1.6722e-02,
                5.1903e-02,
                1.0949e-01,
                2.5017e-01,
                1.5021e00,
                1.0,
            ],
            [0.5, 0.5, 500.0, 500.0, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.0],
            [
                1824.0,
                500.0,
                500.0,
                500.0,
                5.4855e-04,
                5.5127e-03,
                1.6722e-02,
                5.1903e-02,
                1.0949e-01,
                2.5017e-01,
                1.5021e00,
                0.0,
            ],
            [-1] * 12,
            [-1] * 12,
        ]
    ).to(torch.float64)

    # Act.
    _, output = example_policy.generate_input_tensor(
        example_policy.market,
        example_policy.portfolio,
        example_policy.n_stocks_to_sample,
    )

    # Assert.
    assert torch.allclose(output, expected_result, atol=1e-4)
