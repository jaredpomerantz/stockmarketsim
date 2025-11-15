"""A module for the Perceptron class."""

import torch

from simulator.objects.policies.architectures import ModelTask
from simulator.objects.policies.architectures.base_nn import BaseNN


class MultiLayerPerceptron(BaseNN):
    """The class definition for the MultiLayerPerceptron."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: list[int],
        n_classes: int,
        model_task: ModelTask,
    ) -> None:
        """Initializes the MultiLayerPerceptron class."""
        super().__init__()
        layers: list[torch.nn.Module] = []
        n_current_channels = in_channels
        for hidden_channel in hidden_channels:
            layers.append(torch.nn.Linear(n_current_channels, hidden_channel))
            layers.append(torch.nn.ReLU())
            n_current_channels = hidden_channel

        if model_task == ModelTask.CLASSIFIER:
            layers.append(torch.nn.Linear(n_current_channels, n_classes))
            layers.append(torch.nn.Softmax(dim=-1))
        else:
            layers.append(torch.nn.Linear(n_current_channels, 1))

        self.network = torch.nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Runs an input through the network.

        Args:
            input: The tensor to run through the network.

        Returns:
            The model output.

        """
        output = self.network(input).squeeze()
        exp_output = torch.exp(output / 1000)
        return exp_output
