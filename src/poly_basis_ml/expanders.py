"""
ChebyshevExpander: standalone sklearn transformer for Chebyshev polynomial
feature expansion.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_array, check_is_fitted

from ._vandermonde import get_vandermonde_matrix


class ChebyshevExpander(BaseEstimator, TransformerMixin):
    """Chebyshev polynomial feature expansion.

    Maps features to [-1, 1] via MinMaxScaler, then generates a Chebyshev
    Vandermonde matrix with proper intercept handling (one T0 term kept,
    redundant T0 columns stripped).

    Parameters
    ----------
    complexity : int, default=5
        Chebyshev polynomial degree.
    clip_input : bool, default=True
        Clip prediction-time inputs to the training range before scaling.

    Attributes
    ----------
    scaler_ : MinMaxScaler
        Fitted scaler mapping features to [-1, 1].
    n_features_in_ : int
        Number of input features seen during fit.

    Examples
    --------
    >>> from poly_basis_ml import ChebyshevExpander
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.linear_model import Ridge
    >>>
    >>> pipe = Pipeline([('expand', ChebyshevExpander(complexity=5)), ('reg', Ridge())])
    >>> pipe.fit(X_train, y_train)
    """

    def __init__(self, complexity=5, clip_input=True):
        self.complexity = complexity
        self.clip_input = clip_input

    def fit(self, X, y=None):
        """Fit the MinMaxScaler to map features to [-1, 1].

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored

        Returns
        -------
        self
        """
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        self.scaler_ = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_.fit(X)
        return self

    def transform(self, X):
        """Scale to [-1, 1] and generate Chebyshev Vandermonde matrix.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_poly : ndarray of shape (n_samples, n_terms)
            Chebyshev polynomial features.
        """
        check_is_fitted(self)
        X = check_array(X)

        if self.clip_input:
            X = np.clip(X, self.scaler_.data_min_, self.scaler_.data_max_)

        X_scaled = self.scaler_.transform(X)
        return get_vandermonde_matrix(X_scaled, self.complexity)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names.

        Returns
        -------
        list of str
            Names like 'T0_f0', 'T1_f0', ..., 'T1_f1', ...
        """
        check_is_fitted(self)
        d = self.n_features_in_
        c = self.complexity
        names = []
        # First feature keeps T0..Tc
        for deg in range(c + 1):
            names.append(f'T{deg}_f0')
        # Remaining features: T1..Tc (T0 stripped)
        for j in range(1, d):
            for deg in range(1, c + 1):
                names.append(f'T{deg}_f{j}')
        return names
