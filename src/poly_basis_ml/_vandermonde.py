"""
Low-level Chebyshev Vandermonde matrix generation with intercept handling.
"""

import numpy as np
from numpy.polynomial.chebyshev import chebvander


def get_vandermonde_matrix(X, complexity):
    """Generate Chebyshev Vandermonde matrix with intercept handling.

    Keeps T0 from the first feature and strips redundant T0 columns from
    remaining features to avoid perfect multicollinearity.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data, should be scaled to [-1, 1].
    complexity : int
        Chebyshev polynomial degree.

    Returns
    -------
    vander : ndarray of shape (n_samples, n_terms)
        Vandermonde matrix with exactly one intercept term.
    """
    V_first = chebvander(X[:, 0], complexity)
    V_rest = [chebvander(X[:, j], complexity)[:, 1:] for j in range(1, X.shape[1])]
    return np.hstack([V_first] + V_rest)


class _VandermondeTransform:
    """Picklable callable for Vandermonde matrix generation.

    Replaces lambda to enable joblib/pickle serialisation of fitted models.
    """
    def __init__(self, complexity):
        self.complexity = complexity

    def __call__(self, X):
        return get_vandermonde_matrix(X, self.complexity)
