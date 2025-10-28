"""A module for the BaseNN class."""

from abc import abstractmethod

import torch


class BaseNN(torch.nn.Module):
    """The class definition for the MultiLayerPerceptron."""

    def __init__(self) -> None:
        """Instantiates the BaseNN."""
        super().__init__()

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Runs an input through the network.

        Args:
            input: The tensor to run through the network.

        Returns:
            The model output.

        """
        raise NotImplementedError
