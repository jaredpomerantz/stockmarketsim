"""Definition for the Market class."""

import random

import numpy as np

from simulator.objects.orders import BuyOrder, SellOrder
from simulator.objects.participant import Participant
from simulator.objects.stock import Stock, StockHolding

MINIMUM_STOCK_PRICE = 1.0
DAYS_IN_A_QUARTER = 92


class Market:
    """The class definition for the Market.

    This class is responsible for facilitating macro-level market conditions,
    contains all the stocks in the market, moves the market forward, and
    creates events that will effect stocks.
    """

    def __init__(
        self,
        stocks: list[Stock],
        interest_rate_apy: float,
        participants: list[Participant],
    ) -> None:
        """Initialize the Market class.

        Args:
            stocks: The stocks to include in the market.
            interest_rate_apy: The interest rate with which stocks' cash grows.
                This is an annualized percentage yield (APY).
            participants: The participants in the market.

        """
        self.stocks = stocks
        self.interest_rate_apy = interest_rate_apy
        self.participants = participants
        self.participant_id_map = {
            participant.id: participant for participant in self.participants
        }

        self.minimum_stock_price = MINIMUM_STOCK_PRICE

    def advance_stocks(self) -> None:
        """Advances the market's stocks forward one day."""
        stocks_to_delist = []
        for stock in self.stocks:
            stock.price = self.inject_noise_into_stock_price(stock)
            stock.update_price_history(stock.price)
            stock.compound_cash(self.interest_rate_apy)
            if self.check_stock_for_delisting(stock):
                stocks_to_delist.append(stock)

        for stock in stocks_to_delist:
            self.delist_stock(stock)

    def inject_noise_into_stock_price(self, stock: Stock) -> float:
        """Injects random noise into stocks.

        Samples from a 0-mean normal distribution with standard deviation equal
        to the stock's volatility times its price.

        Args:
            stock: The stock object to add noise to.

        Returns:
            An updated stock price.

        """
        updated_price = stock.price + np.random.normal(
            0, stock.stock_volatility * stock.price
        )
        return updated_price

    def check_stock_for_delisting(self, stock: Stock) -> bool:
        """Check if a stock meets the criteria to be delisted.

        A stock will be delisted if any of the following are true:
            Its price dips below the minimum stock price.
            Its quarterly earnings are smaller than the debt that the stock
                accrues in a quarter due to interest. This represents that the
                company that the stock represents can no longer service their
                debt and are going bankrupt.

        Inputs:
            stock: The stock to check for delisting.

        Returns:
            A boolean representing whether the stock is to be delisted.

        """
        if stock.price < self.minimum_stock_price:
            return True
        if stock.cash < 0 and stock.latest_quarterly_earnings < -(
            stock.cash * (self.interest_rate_apy + 1) ** (DAYS_IN_A_QUARTER / 365)
        ):
            return True
        return False

    def delist_stock(self, stock: Stock) -> None:
        """Delists a stock.

        This occurs when a stock dips below the minimum stock price or a stock
        goes bankrupt. In this initial implementation, the stock price will be
        manually set to 0, and the stock will be removed from the market.

        Inputs:
            stock: The stock to delist.
        """
        stock.price = 0.0
        self.stocks.remove(stock)

    def resolve_market_orders(
        self,
        buy_orders: list[tuple[BuyOrder, str]],
        sell_orders: list[tuple[SellOrder, str]],
    ) -> None:
        """Resolves the orders in the market.

        Args:
            buy_orders: A list of tuples containing BuyOrders and the orderer's ID.
            sell_orders: A list of tuples containing SellOrders and the orderer's ID.
        """
        buy_order_stocks = {buy_order[0].stock: [] for buy_order in buy_orders}
        for buy_order in buy_orders:
            buy_order_stocks[buy_order[0].stock] += [
                (buy_order[0].stock, buy_order[0].price, buy_order[1])
                for _ in range(buy_order[0].quantity)
            ]

        sell_order_stocks = {sell_order[0].stock: [] for sell_order in sell_orders}
        for sell_order in sell_orders:
            sell_order_stocks[sell_order[0].stock] += [
                (sell_order[0].stock, sell_order[0].price, sell_order[1])
                for _ in range(sell_order[0].quantity)
            ]

        for buy_order_stock in buy_order_stocks:
            if buy_order_stock not in sell_order_stocks:
                continue

            n_shares_to_transfer = min(
                len(buy_order_stocks[buy_order_stock]),
                len(sell_order_stocks[buy_order_stock]),
            )

            for order_number in range(n_shares_to_transfer):
                # Start with the highest bidding buyer.
                buy_order_instance = buy_order_stocks[buy_order_stock][order_number]

                # Start with the lowest bidding seller.
                sell_order_instance = sell_order_stocks[buy_order_stock][-order_number]

                buyer_id = buy_order_instance[2]
                buyer_participant = self.participant_id_map[buyer_id]
                seller_id = sell_order_instance[2]
                seller_participant = self.participant_id_map[seller_id]
                stock_cost = (buy_order_instance[1] + sell_order_instance[1]) / 2

                if stock_cost > buyer_participant.cash:
                    continue

                stock_holding_to_transfer = StockHolding(
                    stock=buy_order_stock, stock_quantity=1
                )

                buyer_participant.stock_portfolio.add_stock_holding(
                    stock_holding_to_transfer
                )
                buyer_participant.cash -= stock_cost
                seller_participant.stock_portfolio.remove_stock_holding(
                    stock_holding_to_transfer
                )
                seller_participant.cash += stock_cost

    def step_market(self) -> None:
        """Steps the market forward one day."""
        buy_orders: list[tuple[BuyOrder, str]] = []
        sell_orders: list[tuple[SellOrder, str]] = []

        participant_list = self.participants.copy()
        random.shuffle(participant_list)
        for participant in participant_list:
            participant_buy_orders, participant_sell_orders = (
                participant.step_participant()
            )
            buy_orders += participant_buy_orders
            sell_orders += participant_sell_orders

        # Sort the orders by the stock ID, then price.
        buy_orders = sorted(buy_orders, reverse=True)
        sell_orders = sorted(sell_orders, reverse=True)
        self.resolve_market_orders(buy_orders, sell_orders)

        self.advance_stocks()
