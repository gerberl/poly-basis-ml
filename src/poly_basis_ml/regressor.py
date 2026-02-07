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


class ChebyshevRegressor(RegressorMixin, BaseEstimator):
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

    def fit(self, X, y, sample_weight=None):
        """Fit the Chebyshev regression model.

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

        fit_params = {}
        if sample_weight is not None:
            fit_params['reg__sample_weight'] = sample_weight
        self.pipe_.fit(X, y, **fit_params)
        self._build_feature_mapping()
        self.n_outputs_ = 1
        return self

    def _build_feature_mapping(self):
        """Build mappings from input features to polynomial output features."""
        input_names = (list(self.feature_names_in_)
                       if hasattr(self, 'feature_names_in_')
                       else [f'x{i}' for i in range(self.n_features_in_)])

        self._feature_mapping_ = {}
        for idx, name in enumerate(input_names):
            start = 0 if idx == 0 else 1
            self._feature_mapping_[name] = [
                f'{name}_T{deg}' for deg in range(start, self.complexity + 1)
            ]

        self._feature_names_out_ = [
            out for outs in self._feature_mapping_.values() for out in outs
        ]
        self._output_to_input_ = {
            out: inp for inp, outs in self._feature_mapping_.items()
            for out in outs
        }

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

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for the polynomial transform.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input feature names. If provided, rebuilds names from these
            instead of stored feature_names_in_.

        Returns
        -------
        ndarray of str
            Feature names like ['x0_T0', 'x0_T1', ..., 'x1_T1', ...].
        """
        check_is_fitted(self, ['pipe_', '_feature_names_out_'])
        if input_features is not None:
            names = []
            for idx, name in enumerate(input_features):
                start = 0 if idx == 0 else 1
                for deg in range(start, self.complexity + 1):
                    names.append(f'{name}_T{deg}')
            return np.asarray(names)
        return np.asarray(self._feature_names_out_)

    def get_feature_mapping(self):
        """Get mapping from input features to polynomial output features.

        Returns
        -------
        dict
            {input_name: [output_name, ...], ...}
        """
        check_is_fitted(self, ['pipe_', '_feature_mapping_'])
        return self._feature_mapping_.copy()

    def coef_table(self):
        """Return DataFrame of coefficients with feature names.

        Returns
        -------
        pd.DataFrame
            Columns: feature, coef, input_feature. Sorted by |coef| descending.
        """
        import pandas as pd
        check_is_fitted(self, ['pipe_', '_output_to_input_'])
        feature_names = self.get_feature_names_out()
        coefs = self.coef_
        # Only map the main polynomial terms (interactions handled separately)
        n_main = len(feature_names)
        df = pd.DataFrame({
            'feature': feature_names[:len(coefs)] if len(coefs) != n_main else feature_names,
            'coef': coefs[:n_main] if len(coefs) != n_main else coefs,
            'input_feature': [self._output_to_input_.get(f, '?') for f in
                              (feature_names[:len(coefs)] if len(coefs) != n_main else feature_names)],
        })
        return df.sort_values('coef', key=abs, ascending=False).reset_index(drop=True)

    @property
    def feature_importances_(self):
        """Per-input-feature importance (normalised absolute coefficients).

        Excludes the intercept term (T0 of first feature) from aggregation.

        Returns
        -------
        ndarray of shape (n_features_in_,)
        """
        check_is_fitted(self, ['pipe_', '_feature_mapping_'])
        coef_dict = dict(zip(self._feature_names_out_, self.coef_))

        raw = np.array([
            sum(abs(coef_dict[out])
                for out in outs[1 if idx == 0 else 0:])
            for idx, outs in enumerate(self._feature_mapping_.values())
        ])
        total = raw.sum()
        return raw / total if total > 0 else raw

    @property
    def effective_complexity_(self):
        """Effective polynomial complexity per input feature.

        For each input feature, computes the coefficient-weighted mean
        polynomial degree:

            sum(deg * |coef_deg|) / sum(|coef_deg|)

        A value near 1 means the fitted function is approximately linear
        in that feature regardless of the nominal `complexity` setting;
        a value near `complexity` means the model genuinely exploits
        high-order terms. The intercept term (T0) is excluded since it
        carries no directional complexity.

        Returns
        -------
        ndarray of shape (n_features_in_,)
            Effective complexity per input feature. NaN for features
            whose coefficients are all zero.
        """
        check_is_fitted(self, ['pipe_', '_feature_mapping_'])
        coef_dict = dict(zip(self._feature_names_out_, self.coef_))

        result = np.empty(self.n_features_in_)
        for idx, (_, outs) in enumerate(self._feature_mapping_.items()):
            # Skip T0 (intercept) for first feature
            start = 1 if idx == 0 else 0
            terms = outs[start:]
            # Degree of each term: parse from name suffix _T{deg}
            degs = np.array([int(name.rsplit('_T', 1)[1]) for name in terms])
            abs_coefs = np.array([abs(coef_dict[name]) for name in terms])
            total = abs_coefs.sum()
            result[idx] = (degs * abs_coefs).sum() / total if total > 0 else np.nan

        return result


# Backward-compat alias for joblib files pickled under the old name
PolyBasisRegressor = ChebyshevRegressor
