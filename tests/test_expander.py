"""Tests for ChebyshevExpander."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

from poly_basis_ml import ChebyshevExpander


def test_fit_transform_shape():
    """Output shape (n, d*complexity + 1)."""
    rng = np.random.RandomState(0)
    X = rng.randn(100, 3)
    exp = ChebyshevExpander(complexity=4)
    X_out = exp.fit_transform(X)
    # First feature: 5 cols (T0..T4), others: 4 cols each (T1..T4)
    expected_cols = 5 + 2 * 4
    assert X_out.shape == (100, expected_cols)


def test_transform_clipping():
    """Out-of-range inputs handled via clipping."""
    rng = np.random.RandomState(0)
    X_train = rng.randn(100, 2)
    exp = ChebyshevExpander(complexity=3)
    exp.fit(X_train)
    # Test with out-of-range values
    X_test = X_train * 5
    X_out = exp.transform(X_test)
    assert not np.any(np.isnan(X_out))


def test_feature_names_out():
    """Returns T0_f0, T1_f0, ..."""
    rng = np.random.RandomState(0)
    X = rng.randn(50, 2)
    exp = ChebyshevExpander(complexity=3)
    exp.fit(X)
    names = exp.get_feature_names_out()
    assert names[0] == 'T0_f0'
    assert names[1] == 'T1_f0'
    assert 'T1_f1' in names


def test_sklearn_clone():
    exp = ChebyshevExpander(complexity=7)
    cloned = clone(exp)
    assert cloned.complexity == 7


def test_pipeline_integration():
    """Pipeline([ChebyshevExpander(), Ridge()]).fit works."""
    rng = np.random.RandomState(42)
    X = rng.randn(200, 3)
    y = X[:, 0] ** 2 + rng.normal(0, 0.1, 200)
    pipe = Pipeline([('expand', ChebyshevExpander(complexity=4)), ('reg', Ridge())])
    pipe.fit(X, y)
    r2 = pipe.score(X, y)
    assert r2 > 0.5
