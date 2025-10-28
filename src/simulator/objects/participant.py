"""Definition for the Participant class."""

import uuid

from simulator.objects.policies.base_policy import BasePolicy
from simulator.objects.stock import Portfolio


class Participant:
    """The class definition for the Participant.

    Participants own stock and perform trades on the market.
    """

    def __init__(self, stock_portfolio: Portfolio, policy: BasePolicy) -> None:
        """Initialize the Participant class.

        Args:
            stock_portfolio: The stocks owned by the participant.
            policy: The policy that the participant will use.

        """
        self.id = uuid.uuid4()
        self.stock_portfolio = stock_portfolio
        self.policy = policy

    def get_stock_portfolio_value(self) -> float:
        """Get the total value of the Participant's stock portfolio."""
        return sum(
            [
                stock_holding.get_holding_value()
                for stock_holding in self.stock_portfolio.get_stock_holding_list()
            ]
        )

    def update_policy(self, market_baseline: float) -> None:
        """Updates the participant's policy based on performance vs market.

        Args:
            market_baseline: The total market's performance.

        """
        # self.policy.update()
