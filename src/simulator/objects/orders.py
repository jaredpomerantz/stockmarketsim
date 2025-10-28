"""Definition for the classes related to Orders."""

from dataclasses import dataclass

from simulator.objects.stock import Stock


@dataclass
class BuyOrder:
    """The object representing Buy Orders in the market."""

    stock: Stock
    quantity: int
    price: float


@dataclass
class SellOrder:
    """The object representing Sell Orders in the market."""

    stock: Stock
    quantity: int
    price: float
