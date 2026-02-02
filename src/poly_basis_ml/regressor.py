"""
ChebyshevRegressor: convenience estimator wrapping ChebyshevExpander + Ridge.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import r2_score

from ._vandermonde import _VandermondeTransform
from .interactions import _PolyInteractionTransform


class ChebyshevRegressor(BaseEstimator, RegressorMixin):
    """Chebyshev polynomial regression with Ridge regularisation.

    Internally builds Pipeline(MinMaxScaler, ChebyshevVandermonde, Ridge).
    Optionally adds bivariate interaction features.

    Parameters
    ----------
    complexity : int, default=5
        Chebyshev polynomial degree.
    alpha : float, default=1.0
        Ridge regularisation strength.
    clip_input : bool, default=True
        Clip prediction-time inputs to training range.
    include_interactions : bool, default=False
        Whether to generate interaction features.
    interaction_types : list of str or None
        Types of interactions (default: ['product']).
    interaction_pairs : str, default='auto'
        Which pairs: 'auto', 'all', 'top_n', 'none'.
    max_interactions : int, default=100
        Hard limit on interaction pairs.
    expand_interactions : bool, default=False
        Apply Chebyshev expansion to interactions.
    max_interaction_complexity : int, default=5
        Max degree for interaction expansion.
    interaction_ranking : str, default='variance'
        Feature ranking: 'variance' or 'mi_rescue'.

    Examples
    --------
    >>> from poly_basis_ml import ChebyshevRegressor
    >>> model = ChebyshevRegressor(complexity=8, alpha=0.1)
    >>> model.fit(X_train, y_train)
    >>> print(f"R2: {model.score(X_test, y_test):.3f}")
    """

    def __init__(self, complexity=5, alpha=1.0, clip_input=True,
                 include_interactions=False, interaction_types=None,
                 interaction_pairs='auto', max_interactions=100,
                 expand_interactions=False, max_interaction_complexity=5,
                 interaction_ranking='variance'):
        self.complexity = complexity
        self.alpha = alpha
        self.clip_input = clip_input
        self.include_interactions = include_interactions
        self.interaction_types = interaction_types
        self.interaction_pairs = interaction_pairs
        self.max_interactions = max_interactions
        self.expand_interactions = expand_interactions
        self.max_interaction_complexity = max_interaction_complexity
        self.interaction_ranking = interaction_ranking

    def fit(self, X, y):
        """Fit the Chebyshev regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        if self.include_interactions:
            transform = _PolyInteractionTransform(
                complexity=self.complexity,
                include_interactions=True,
                interaction_types=self.interaction_types or ['product'],
                interaction_pairs=self.interaction_pairs,
                max_interactions=self.max_interactions,
                expand_interactions=self.expand_interactions,
                max_interaction_complexity=self.max_interaction_complexity,
                interaction_ranking=self.interaction_ranking,
            )
            self.pipe_ = Pipeline([
                ('scl', MinMaxScaler(feature_range=(-1, 1))),
                ('vdr', transform),
                ('reg', Ridge(alpha=self.alpha, fit_intercept=False)),
            ])
        else:
            transform = _VandermondeTransform(self.complexity)
            self.pipe_ = Pipeline([
                ('scl', MinMaxScaler(feature_range=(-1, 1))),
                ('vdr', FunctionTransformer(transform)),
                ('reg', Ridge(alpha=self.alpha, fit_intercept=False)),
            ])

        self.pipe_.fit(X, y)
        return self

    def predict(self, X):
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        check_is_fitted(self, 'pipe_')
        X = check_array(X)
        if self.clip_input:
            scaler = self.pipe_['scl']
            X = np.clip(X, scaler.data_min_, scaler.data_max_)
        return self.pipe_.predict(X)

    @property
    def coef_(self):
        check_is_fitted(self, 'pipe_')
        return self.pipe_['reg'].coef_

    @property
    def intercept_(self):
        check_is_fitted(self, 'pipe_')
        return self.pipe_['reg'].intercept_
