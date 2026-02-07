"""Tests for ChebyshevRegressor."""

import numpy as np
from poly_basis_ml import ChebyshevRegressor


def test_fit_predict_1d_sin():
    """sin(x), R2 > 0.95 with complexity=8."""
    rng = np.random.RandomState(42)
    X = rng.uniform(-3, 3, (300, 1))
    y = np.sin(X[:, 0]) + rng.normal(0, 0.05, 300)
    model = ChebyshevRegressor(complexity=8, alpha=0.01)
    model.fit(X, y)
    r2 = model.score(X, y)
    assert r2 > 0.95, f"R2={r2:.3f}"


def test_fit_predict_nd_friedman1():
    """friedman1, R2 > 0.5."""
    from sklearn.datasets import make_friedman1
    X, y = make_friedman1(n_samples=500, n_features=5, random_state=42)
    model = ChebyshevRegressor(complexity=3, alpha=1.0)
    model.fit(X, y)
    r2 = model.score(X, y)
    assert r2 > 0.5, f"R2={r2:.3f}"


def test_with_interactions():
    """include_interactions=True runs without error."""
    rng = np.random.RandomState(42)
    X = rng.randn(200, 4)
    y = X[:, 0] * X[:, 1] + rng.normal(0, 0.1, 200)
    model = ChebyshevRegressor(complexity=3, include_interactions=True)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (200,)


def test_get_set_params():
    model = ChebyshevRegressor(complexity=5, alpha=0.5)
    params = model.get_params()
    assert params['complexity'] == 5
    assert params['alpha'] == 0.5
    model.set_params(complexity=3)
    assert model.complexity == 3
