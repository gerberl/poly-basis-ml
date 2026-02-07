"""
Bivariate interaction functions for feature engineering.

Three families (pick one from each to avoid redundancy):
- Product family: product, harmonic (multiplicative relationships)
- Ratio family: contrast, ratio, log_ratio (relative comparisons)
- Additive family: addition, difference (linear combinations)
"""

import numpy as np
from itertools import combinations

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array

from ._vandermonde import get_vandermonde_matrix


# --- Interaction functions ---

def _interaction_product(a, b, eps=1e-6):
    """Product interaction: a * b."""
    return a * b

def _interaction_harmonic(a, b, eps=1e-6):
    """Harmonic interaction: 2*a*b / (|a| + |b| + eps)."""
    return 2 * a * b / (np.abs(a) + np.abs(b) + eps)

def _interaction_contrast(a, b, eps=1e-6):
    """Contrast interaction: (a - b) / (|a| + |b| + eps)."""
    return (a - b) / (np.abs(a) + np.abs(b) + eps)

def _interaction_ratio(a, b, eps=1e-6):
    """Ratio interaction: a / (b + eps). Unbounded."""
    return a / (b + eps)

def _interaction_log_ratio(a, b, eps=1e-6):
    """Log-ratio interaction: log(|a| + eps) - log(|b| + eps)."""
    return np.log(np.abs(a) + eps) - np.log(np.abs(b) + eps)

def _interaction_addition(a, b, eps=1e-6):
    """Addition interaction: (a + b) / 2."""
    return (a + b) / 2

def _interaction_difference(a, b, eps=1e-6):
    """Difference interaction: (a - b) / 2."""
    return (a - b) / 2


INTERACTION_FUNCS = {
    'product': _interaction_product,
    'harmonic': _interaction_harmonic,
    'contrast': _interaction_contrast,
    'ratio': _interaction_ratio,
    'log_ratio': _interaction_log_ratio,
    'addition': _interaction_addition,
    'difference': _interaction_difference,
}

RECOMMENDED_INTERACTION_TYPES = ['product', 'contrast', 'addition']


class _PolyInteractionTransform(BaseEstimator):
    """Sklearn transformer for combined Chebyshev polynomial + interaction features.

    Generates Chebyshev Vandermonde matrix and optionally adds interaction features.
    Supports variance-based and MI-based feature ranking for pair selection.
    """

    def __init__(self, complexity,
                 include_interactions=False, interaction_types=None,
                 interaction_pairs='auto', interaction_d_threshold=30,
                 interaction_top_frac=0.5, interaction_top_n=None,
                 max_interactions=100,
                 expand_interactions=False, max_interaction_complexity=5,
                 interaction_ranking='variance', mi_spearman_threshold=0.1):
        self.complexity = complexity
        self.include_interactions = include_interactions
        self.interaction_types = interaction_types or ['product']
        self.interaction_pairs = interaction_pairs
        self.interaction_d_threshold = interaction_d_threshold
        self.interaction_top_frac = interaction_top_frac
        self.interaction_top_n = interaction_top_n
        self.max_interactions = max_interactions
        self.expand_interactions = expand_interactions
        self.max_interaction_complexity = max_interaction_complexity
        self.interaction_ranking = interaction_ranking
        self.mi_spearman_threshold = mi_spearman_threshold
        self._interaction_pairs_cache = None
        self._feature_ranking = None

    def _rank_features_variance(self, X):
        variances = np.var(X, axis=0)
        return np.argsort(variances)[::-1]

    def _rank_features_mi_rescue(self, X, y):
        from scipy.stats import spearmanr
        from sklearn.feature_selection import mutual_info_regression

        n_features = X.shape[1]
        spearman_scores = np.array([
            abs(spearmanr(X[:, i], y, nan_policy='omit')[0])
            for i in range(n_features)
        ])
        spearman_scores = np.nan_to_num(spearman_scores, nan=0.0)

        low_spearman_mask = spearman_scores < self.mi_spearman_threshold
        mi_scores = np.zeros(n_features)
        if low_spearman_mask.any():
            rescue_indices = np.where(low_spearman_mask)[0]
            mi_rescue = mutual_info_regression(
                X[:, rescue_indices], y, random_state=42
            )
            mi_scores[rescue_indices] = mi_rescue

        spearman_max = spearman_scores.max()
        spearman_norm = spearman_scores / (spearman_max + 1e-10) if spearman_max > 0 else spearman_scores
        mi_max = mi_scores.max()
        mi_norm = mi_scores / (mi_max + 1e-10) if mi_max > 0 else mi_scores

        combined_scores = np.where(
            low_spearman_mask & (mi_norm > spearman_norm),
            mi_norm, spearman_norm
        )
        return np.argsort(combined_scores)[::-1]

    def _compute_feature_ranking(self, X, y=None):
        if self.interaction_ranking == 'mi_rescue' and y is not None:
            return self._rank_features_mi_rescue(X, y)
        return self._rank_features_variance(X)

    def _get_pairs_from_ranking(self, X, ranking):
        n_features = X.shape[1]

        if not self.include_interactions or self.interaction_pairs == 'none':
            return []

        if self.interaction_pairs == 'auto':
            if n_features <= self.interaction_d_threshold:
                pairs = list(combinations(range(n_features), 2))
            else:
                n_top = max(2, int(n_features * self.interaction_top_frac))
                top_indices = ranking[:n_top]
                pairs = list(combinations(sorted(top_indices), 2))
        elif self.interaction_pairs == 'all':
            pairs = list(combinations(range(n_features), 2))
        elif self.interaction_pairs == 'top_n':
            n = self.interaction_top_n or min(n_features, 10)
            top_indices = ranking[:n]
            pairs = list(combinations(sorted(top_indices), 2))
        else:
            pairs = list(combinations(range(n_features), 2))

        if self.max_interactions is not None:
            n_types = len(self.interaction_types)
            max_pairs = self.max_interactions // max(1, n_types)
            if len(pairs) > max_pairs:
                rank_lookup = {idx: rank for rank, idx in enumerate(ranking)}
                pair_scores = [
                    (i, j, rank_lookup.get(i, n_features) + rank_lookup.get(j, n_features))
                    for i, j in pairs
                ]
                pair_scores.sort(key=lambda x: x[2])
                pairs = [(p[0], p[1]) for p in pair_scores[:max_pairs]]

        return pairs

    def fit(self, X, y=None):
        X = check_array(X)
        self._feature_ranking = self._compute_feature_ranking(X, y)
        self._interaction_pairs_cache = self._get_pairs_from_ranking(X, self._feature_ranking)
        return self

    def transform(self, X):
        X = check_array(X)
        X_poly = get_vandermonde_matrix(X, self.complexity)

        if self.include_interactions and self._interaction_pairs_cache:
            interaction_blocks = []
            for i, j in self._interaction_pairs_cache:
                for itype in self.interaction_types:
                    func = INTERACTION_FUNCS[itype]
                    z = func(X[:, i], X[:, j])

                    if self.expand_interactions:
                        z_clipped = np.clip(z, -1, 1).reshape(-1, 1)
                        inter_complexity = min(self.complexity, self.max_interaction_complexity)
                        z_poly = get_vandermonde_matrix(z_clipped, inter_complexity)
                        interaction_blocks.append(z_poly[:, 1:])
                    else:
                        interaction_blocks.append(z.reshape(-1, 1))

            if interaction_blocks:
                X_inter = np.hstack(interaction_blocks)
                return np.hstack([X_poly, X_inter])

        return X_poly

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
