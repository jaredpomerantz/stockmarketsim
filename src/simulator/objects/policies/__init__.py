"""A module for Policies."""

from sklearn.base import BaseEstimator, RegressorMixin


class MLRegressor(RegressorMixin, BaseEstimator):
    """Regressor class for sklearn regressors."""
