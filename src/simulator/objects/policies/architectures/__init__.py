"""A module for neural network architectures."""

from enum import Enum


class ModelTask(Enum):
    """Enum for model task."""

    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"
