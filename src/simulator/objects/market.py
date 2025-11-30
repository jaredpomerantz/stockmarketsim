"""Definition for the Market class."""

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
        self.participants: list[Participant] = []
        self.market_value_history = np.ones(shape=(1825,)) * self.get_market_value()

        self.minimum_stock_price = MINIMUM_STOCK_PRICE

    def add_participants(self, participants: list[Participant]):
        """Adds the participant list to the market.

        Participant policy instatiation requires the market to already be instantiated
        Therefore, they must be added after the market initialization.

        Args:
            participants: The participants in the market.
        """
        self.participants += participants
        self.participant_id_map = {
            participant.id: participant for participant in self.participants
        }

    def advance_stocks(self) -> None:
        """Advances the market's stocks forward one day."""
        stocks_to_delist = []
        for stock in self.stocks:
            if self.check_stock_for_delisting(stock):
                stocks_to_delist.append(stock)
                continue
            stock.price = self.inject_noise_into_stock_price(stock)
            stock.compound_cash(self.interest_rate_apy)
            stock.step()

        for stock in stocks_to_delist:
            print(f"delisting stock: {stock}")
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
            stock.cash * (self.interest_rate_apy / 4)
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
        stock.update_price_history(stock.price)
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
        buy_order_stocks: dict[Stock, list[tuple[Stock, float, str]]] = {
            buy_order[0].stock: [] for buy_order in buy_orders
        }
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

        print(f"Number of buy order stocks: {len(buy_order_stocks.keys())}")
        print(f"Number of sell order stocks: {len(sell_order_stocks.keys())}")
        print(f"Market total value: {self.get_market_value()}")

        for sell_order_stock in sell_order_stocks:
            if sell_order_stock not in buy_order_stocks:
                # Case that participants want to sell a stock but nobody is buying:
                # decrease the stock price. Move stock price 10% toward the lowest bid,
                # or if the bids were above the current price, decrease the stock price
                # by 5%.
                sell_order_stock.price = sell_order_stock.price + min(
                    (sell_order_stocks[sell_order_stock][0][1] - sell_order_stock.price)
                    * 0.1,
                    sell_order_stock.price * 0.05,
                )
                continue

        for buy_order_stock in buy_order_stocks:
            if buy_order_stock not in sell_order_stocks:
                # Case that participants want to buy a stock but nobody is selling:
                # increase the stock price. Move stock price 10% toward the highest bid,
                # or if the bids were below the current price, increase the stock price
                # by 5%.
                buy_order_stock.price = buy_order_stock.price + max(
                    (buy_order_stocks[buy_order_stock][0][1] - buy_order_stock.price)
                    * 0.1,
                    buy_order_stock.price * 0.05,
                )
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
                stock_holding_to_transfer.stock.price = stock_cost

    def get_market_value(self) -> float:
        """Gets the total value of all stocks."""
        return sum([stock.price for stock in self.stocks])

    def update_market_value_history(self) -> None:
        """Updates the market value history with the current value."""
        self.market_value_history = np.append(
            self.market_value_history[1:],
            [self.get_market_value()],
        )

    def step_market(self) -> None:
        """Steps the market forward one day."""
        buy_orders: list[tuple[BuyOrder, str]] = []
        sell_orders: list[tuple[SellOrder, str]] = []

        for participant in self.participants:
            participant_buy_orders, participant_sell_orders = (
                participant.step_participant()
            )
            buy_orders += participant_buy_orders
            sell_orders += participant_sell_orders

        # Sort the orders by the stock ID, then price.
        buy_orders = sorted(buy_orders, reverse=True)
        sell_orders = sorted(sell_orders, reverse=True)
        self.resolve_market_orders(buy_orders, sell_orders)
        self.update_market_value_history()

        self.advance_stocks()
