"""
ChebyshevModelTreeRegressor: decision tree with Chebyshev polynomial leaf models.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import r2_score

from .regressor import ChebyshevRegressor


class ChebyshevModelTreeRegressor(RegressorMixin, BaseEstimator):
    """Decision tree with Chebyshev polynomial leaf models.

    The routing tree partitions the feature space into regions, then each
    leaf fits a ChebyshevRegressor for local approximation.

    Parameters
    ----------
    max_depth : int, default=3
        Maximum depth of the routing tree.
    min_samples_leaf : int or float, default=200
        Minimum samples per leaf for fitting a polynomial model.
        Leaves with fewer samples fall back to tree prediction.
        If int, the absolute minimum number of samples.
        If float, a fraction of n_samples (consistent with sklearn).
    complexity : int, default=2
        Chebyshev polynomial degree for leaf models.
    alpha : float, default=10.0
        Ridge regularisation strength for leaf models.
    routing_features : list of int or None, default=None
        Column indices for tree routing. If None, uses all.
    leaf_features : list of int or None, default=None
        Column indices for leaf models. If None, uses all.
    random_state : int, default=42
        Random state for reproducibility.

    Examples
    --------
    >>> from poly_basis_ml import ChebyshevModelTreeRegressor
    >>> model = ChebyshevModelTreeRegressor(max_depth=3, complexity=2)
    >>> model.fit(X_train, y_train)
    >>> print(f"R2: {model.score(X_test, y_test):.3f}")
    """

    def __init__(self, max_depth=3, min_samples_leaf=200,
                 complexity=2, alpha=10.0,
                 routing_features=None, leaf_features=None,
                 random_state=42):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.complexity = complexity
        self.alpha = alpha
        self.routing_features = routing_features
        self.leaf_features = leaf_features
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        """Fit model tree with Chebyshev leaf models.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

        Returns
        -------
        self
        """
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns)
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        n_samples = X.shape[0]

        # Resolve min_samples_leaf: float means fraction of n_samples
        if isinstance(self.min_samples_leaf, float):
            min_leaf_abs = max(1, int(np.ceil(self.min_samples_leaf * n_samples)))
        else:
            min_leaf_abs = self.min_samples_leaf

        X_route = X[:, self.routing_features] if self.routing_features is not None else X
        X_leaf = X[:, self.leaf_features] if self.leaf_features is not None else X

        # Fit routing tree (sklearn handles int/float min_samples_leaf natively)
        self.tree_ = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        tree_fit_params = {}
        if sample_weight is not None:
            tree_fit_params['sample_weight'] = sample_weight
        self.tree_.fit(X_route, y, **tree_fit_params)

        # Fit leaf models
        leaf_ids = self.tree_.apply(X_route)
        self.leaf_models_ = {}
        for leaf_id in np.unique(leaf_ids):
            mask = leaf_ids == leaf_id
            if mask.sum() >= min_leaf_abs:
                leaf_sw = sample_weight[mask] if sample_weight is not None else None
                self.leaf_models_[leaf_id] = ChebyshevRegressor(
                    complexity=self.complexity, alpha=self.alpha,
                ).fit(X_leaf[mask], y[mask], sample_weight=leaf_sw)

        self.n_outputs_ = 1
        return self

    def predict(self, X):
        """Predict using fitted model tree.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        check_is_fitted(self, ['tree_', 'leaf_models_'])
        X = check_array(X)

        X_route = X[:, self.routing_features] if self.routing_features is not None else X
        X_leaf = X[:, self.leaf_features] if self.leaf_features is not None else X

        leaf_ids = self.tree_.apply(X_route)
        y_pred = np.full(len(X), np.nan)

        for leaf_id, model in self.leaf_models_.items():
            mask = leaf_ids == leaf_id
            if mask.any():
                y_pred[mask] = model.predict(X_leaf[mask])

        # Fallback for leaves without polynomial model
        nan_mask = np.isnan(y_pred)
        if nan_mask.any():
            y_pred[nan_mask] = self.tree_.predict(X_route[nan_mask])

        return y_pred


# Backward-compat alias for joblib files pickled under the old name
PolyBasisModelTreeRegressor = ChebyshevModelTreeRegressor
