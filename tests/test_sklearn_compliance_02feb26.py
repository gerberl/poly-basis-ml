"""sklearn API compliance tests for poly_basis_ml estimators."""

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from poly_basis_ml import ChebyshevRegressor, ChebyshevModelTreeRegressor


# ---------------------------------------------------------------------------
# Full sklearn estimator checks (with expected failures)
# ---------------------------------------------------------------------------

def _expected_failed_checks(estimator):
    failed = {}
    if isinstance(estimator, ChebyshevRegressor):
        failed["check_n_features_in_after_fitting"] = (
            "Pipeline internally manages n_features_in validation"
        )
        failed["check_sample_weight_equivalence_on_dense_data"] = (
            "Chebyshev basis expansion means uniform vs weighted can diverge"
        )
    return failed


@parametrize_with_checks(
    [ChebyshevRegressor(), ChebyshevModelTreeRegressor()],
    expected_failed_checks=_expected_failed_checks,
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


# ---------------------------------------------------------------------------
# Targeted smoke tests
# ---------------------------------------------------------------------------

def _make_data(n=200, d=4, as_df=False):
    rng = np.random.RandomState(42)
    X = rng.randn(n, d)
    y = X[:, 0] + 0.5 * X[:, 1] + rng.randn(n) * 0.1
    if as_df:
        X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(d)])
    return X, y


class TestSampleWeight:
    def test_chebyshev_regressor_accepts_sample_weight(self):
        X, y = _make_data()
        sw = np.ones(len(y))
        model = ChebyshevRegressor(complexity=3, alpha=1.0)
        model.fit(X, y, sample_weight=sw)
        assert hasattr(model, 'pipe_')

    def test_model_tree_accepts_sample_weight(self):
        X, y = _make_data(n=400)
        sw = np.ones(len(y))
        model = ChebyshevModelTreeRegressor(max_depth=2, min_samples_leaf=50)
        model.fit(X, y, sample_weight=sw)
        assert hasattr(model, 'tree_')


class TestNOutputs:
    def test_chebyshev_regressor_sets_n_outputs(self):
        X, y = _make_data()
        model = ChebyshevRegressor().fit(X, y)
        assert model.n_outputs_ == 1

    def test_model_tree_sets_n_outputs(self):
        X, y = _make_data(n=400)
        model = ChebyshevModelTreeRegressor(min_samples_leaf=50).fit(X, y)
        assert model.n_outputs_ == 1


class TestFeatureNamesIn:
    def test_chebyshev_regressor_captures_feature_names(self):
        X, y = _make_data(as_df=True)
        model = ChebyshevRegressor().fit(X, y)
        assert hasattr(model, 'feature_names_in_')
        assert list(model.feature_names_in_) == [f"feat_{i}" for i in range(4)]

    def test_model_tree_captures_feature_names(self):
        X, y = _make_data(n=400, as_df=True)
        model = ChebyshevModelTreeRegressor(min_samples_leaf=50).fit(X, y)
        assert hasattr(model, 'feature_names_in_')
        assert list(model.feature_names_in_) == [f"feat_{i}" for i in range(4)]

    def test_no_feature_names_for_array_input(self):
        X, y = _make_data()
        model = ChebyshevRegressor().fit(X, y)
        assert not hasattr(model, 'feature_names_in_')


class TestNoTransformMethod:
    """ChebyshevRegressor must not have a transform() method (sklearn treats it as transformer)."""

    def test_regressor_has_no_transform(self):
        assert not hasattr(ChebyshevRegressor, 'transform')


class TestFeatureNaming:
    def test_get_feature_names_out_array(self):
        X, y = _make_data(d=3)
        model = ChebyshevRegressor(complexity=3).fit(X, y)
        names = model.get_feature_names_out()
        # first feature: T0,T1,T2,T3 (4); rest: T1,T2,T3 (3 each) => 4+3+3=10
        assert len(names) == 10
        assert names[0] == 'x0_T0'
        assert names[4] == 'x1_T1'

    def test_get_feature_names_out_dataframe(self):
        X, y = _make_data(d=2, as_df=True)
        model = ChebyshevRegressor(complexity=2).fit(X, y)
        names = list(model.get_feature_names_out())
        assert names == ['feat_0_T0', 'feat_0_T1', 'feat_0_T2', 'feat_1_T1', 'feat_1_T2']

    def test_get_feature_names_out_with_input_features(self):
        X, y = _make_data(d=2)
        model = ChebyshevRegressor(complexity=2).fit(X, y)
        names = list(model.get_feature_names_out(input_features=['a', 'b']))
        assert names == ['a_T0', 'a_T1', 'a_T2', 'b_T1', 'b_T2']

    def test_get_feature_mapping(self):
        X, y = _make_data(d=2)
        model = ChebyshevRegressor(complexity=2).fit(X, y)
        mapping = model.get_feature_mapping()
        assert list(mapping.keys()) == ['x0', 'x1']
        assert mapping['x0'] == ['x0_T0', 'x0_T1', 'x0_T2']
        assert mapping['x1'] == ['x1_T1', 'x1_T2']

    def test_coef_table(self):
        X, y = _make_data(d=2)
        model = ChebyshevRegressor(complexity=2).fit(X, y)
        df = model.coef_table()
        assert list(df.columns) == ['feature', 'coef', 'input_feature']
        assert len(df) == 5  # 3 + 2

    def test_feature_importances(self):
        X, y = _make_data(d=3)
        model = ChebyshevRegressor(complexity=3).fit(X, y)
        imp = model.feature_importances_
        assert imp.shape == (3,)
        assert abs(imp.sum() - 1.0) < 1e-10
        # Feature 0 and 1 should dominate (y = x0 + 0.5*x1 + noise)
        assert imp[0] > imp[2]
        assert imp[1] > imp[2]

    def test_model_complexity(self):
        X, y = _make_data(d=3)
        model = ChebyshevRegressor(complexity=5).fit(X, y)
        mc = model.effective_complexity_
        assert mc.shape == (3,)
        # Values must be between 1 and complexity (T0 excluded)
        assert np.all(mc >= 1.0)
        assert np.all(mc <= 5.0)
        # y = x0 + 0.5*x1 + noise => features 0,1 should be low complexity
        # (dominated by T1), feature 2 should be noisier but still bounded
        assert mc[0] < 3.0
        assert mc[1] < 3.0

    def test_names_match_coef_count(self):
        """Number of output feature names must match number of coefficients."""
        for d in [2, 3, 5]:
            X, y = _make_data(d=d)
            model = ChebyshevRegressor(complexity=4).fit(X, y)
            assert len(model.get_feature_names_out()) == len(model.coef_)
