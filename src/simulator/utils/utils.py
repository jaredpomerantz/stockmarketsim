"""A utils module for running the market."""

import numpy as np
from numpy.random import default_rng

from simulator.objects.market import Market
from simulator.objects.participant import Participant
from simulator.objects.policies.base_policy import BasePolicy
from simulator.objects.stock import Portfolio, Stock, StockHolding


def generate_portfolio(stocks: list[Stock]) -> Portfolio:
    """Generates a portfolio given a list of Stock objects.

    Samples random numbers of stocks for the portfolio.

    Args:
        stocks: The stock objects to sample into the Portfolio.

    Returns:
        A Portfolio object consisting of StockHoldings for the given Stocks.
    """
    rng = default_rng()
    stock_holdings: list[StockHolding] = []
    for stock in stocks:
        stock_quantity: int = rng.choice(a=np.array([0, 0, 0, 1, 2, 3]), size=1)[0]
        if stock_quantity > 0:
            stock_holdings.append(
                StockHolding(
                    stock=stock,
                    stock_quantity=stock_quantity,
                )
            )
    return Portfolio(stock_holdings=stock_holdings)


def generate_stocks(n_stocks: int) -> list[Stock]:
    """Generates stocks of random parameters.

    Args:
        n_stocks: The number of stocks to generate.

    Returns:
        A list of Stock object.
    """
    rng = default_rng()
    output: list[Stock] = []
    for _ in range(n_stocks):
        output.append(
            Stock(
                cash=rng.uniform(-10000, 30000, size=1)[0],
                earning_value_of_assets=rng.uniform(10000, 30000, size=1)[0],
                latest_quarterly_earnings=rng.uniform(10000, 30000, size=1)[0],
                price_history=np.ones(shape=(1825,)) * rng.uniform(10, 200, size=1)[0],
                quality_of_leadership=rng.uniform(0, 1, size=1)[0],
                stock_volatility=rng.uniform(0, 0.1, size=1)[0],
            )
        )
    return output


def generate_participants(
    n_participants: int,
    stock_list: list[Stock],
    policy_class: type[BasePolicy],
    policy_args: list,
) -> list[Participant]:
    """Generates Participant objects.

    Args:
        n_participants: The number of Participants to generate.
        stock_list: A list of Stock objects to populate the Participants with.
        policy_class: The class of Policy that the Participant should use.
        policy_args: The arguments to instantiate the Policy with.

    Returns:
        A list of Participant objects.
            Will contain initialized Portfolios and Policies.
    """
    output: list[Participant] = []
    for _ in range(n_participants):
        output.append(
            Participant(
                stock_portfolio=generate_portfolio(stock_list),
                policy=policy_class(*policy_args),
                cash=3000,
            )
        )
    return output
