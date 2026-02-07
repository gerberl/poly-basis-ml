"""Tests for ChebyshevModelTreeRegressor."""

import numpy as np
from poly_basis_ml import ChebyshevModelTreeRegressor


def test_fit_predict_basic():
    """Synthetic data, R2 > 0."""
    rng = np.random.RandomState(42)
    X = rng.randn(500, 3)
    y = np.where(X[:, 0] > 0, X[:, 1] ** 2, -X[:, 1]) + rng.normal(0, 0.1, 500)
    model = ChebyshevModelTreeRegressor(max_depth=2, min_samples_leaf=50, complexity=2)
    model.fit(X, y)
    r2 = model.score(X, y)
    assert r2 > 0, f"R2={r2:.3f}"


def test_min_samples_leaf_fallback():
    """Tiny leaf falls back to tree prediction."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 2)
    y = X[:, 0] + rng.normal(0, 0.1, 100)
    # Very high min_samples_leaf so most leaves fall back
    model = ChebyshevModelTreeRegressor(max_depth=5, min_samples_leaf=80)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (100,)
    assert not np.any(np.isnan(pred))


def test_get_set_params():
    model = ChebyshevModelTreeRegressor(max_depth=4, complexity=3)
    params = model.get_params()
    assert params['max_depth'] == 4
    assert params['complexity'] == 3
    model.set_params(max_depth=2)
    assert model.max_depth == 2
