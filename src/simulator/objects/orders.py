"""Definition for the classes related to Orders."""

from dataclasses import dataclass

from simulator.objects.stock import Stock


@dataclass
class BuyOrder:
    """The object representing Buy Orders in the market."""

    stock: Stock
    quantity: int
    price: float

    def __eq__(self, other) -> bool:
        """Defines equality condition for StockHolding objects.

        Args:
            other: The other StockHolding object to compare.
        """
        return self.stock == other.stock and self.quantity == other.quantity

    def __lt__(self, other) -> bool:
        """Defines less-than condition for StockHolding objects.

        Args:
            other: The other StockHolding object to compare.
        """
        return self.stock < other.stock

    def __le__(self, other) -> bool:
        """Defines less-than-or-equal condition for StockHolding objects.

        Args:
            other: The other StockHolding object to compare.
        """
        return self.stock <= other.stock

    def __gt__(self, other) -> bool:
        """Defines greater-than condition for StockHolding objects.

        Args:
            other: The other StockHolding object to compare.
        """
        return self.stock > other.stock

    def __ge__(self, other) -> bool:
        """Defines greater-than-or-equal condition for StockHolding objects.

        Args:
            other: The other StockHolding object to compare.
        """
        return self.stock >= other.stock


@dataclass
class SellOrder:
    """The object representing Sell Orders in the market."""

    stock: Stock
    quantity: int
    price: float

    def __eq__(self, other) -> bool:
        """Defines equality condition for StockHolding objects.

        Args:
            other: The other StockHolding object to compare.
        """
        return self.stock == other.stock and self.quantity == other.quantity

    def __lt__(self, other) -> bool:
        """Defines less-than condition for StockHolding objects.

        Args:
            other: The other StockHolding object to compare.
        """
        return self.stock < other.stock

    def __le__(self, other) -> bool:
        """Defines less-than-or-equal condition for StockHolding objects.

        Args:
            other: The other StockHolding object to compare.
        """
        return self.stock <= other.stock

    def __gt__(self, other) -> bool:
        """Defines greater-than condition for StockHolding objects.

        Args:
            other: The other StockHolding object to compare.
        """
        return self.stock > other.stock

    def __ge__(self, other) -> bool:
        """Defines greater-than-or-equal condition for StockHolding objects.

        Args:
            other: The other StockHolding object to compare.
        """
        return self.stock >= other.stock
