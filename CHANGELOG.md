# Changelog

## 0.2.0 (2026-02-16)

Feature introspection and sklearn API compliance improvements.

- `feature_importances_` property: normalised absolute coefficients per input feature
- `effective_complexity_` property: coefficient-weighted mean polynomial degree per feature
- `get_feature_names_out()`, `get_feature_mapping()`, `coef_table()` methods
- `sample_weight` support in `fit()` for both `ChebyshevRegressor` and `ChebyshevModelTreeRegressor`
- `feature_names_in_` set automatically from DataFrame column names
- MRO fix: `RegressorMixin` before `BaseEstimator` for correct sklearn compliance
- Backward-compatibility aliases: `PolyBasisRegressor`, `PolyBasisModelTreeRegressor`
- Added `pandas>=1.3` dependency
- Added sklearn compliance test suite

## 0.1.0 (2026-02-07)

Initial public release accompanying the paper:
"Revisiting Chebyshev Polynomial and Anisotropic RBF Models
for Tabular Regression" (Gerber & Lloyd, 2026).
