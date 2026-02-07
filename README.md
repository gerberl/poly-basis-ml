# poly-basis-ml

Chebyshev polynomial feature expansion and regression for scikit-learn.

[![PyPI version](https://img.shields.io/pypi/v/poly-basis-ml.svg)](https://pypi.org/project/poly-basis-ml/)
[![Tests](https://github.com/gerberl/poly-basis-ml/actions/workflows/test.yml/badge.svg)](https://github.com/gerberl/poly-basis-ml/actions)

## Installation

```bash
pip install poly-basis-ml
```

## Quick start

```python
from poly_basis_ml import ChebyshevRegressor
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split

X, y = make_friedman1(n_samples=1000, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = ChebyshevRegressor(complexity=5, alpha=0.1)
model.fit(X_train, y_train)
print(f"R2: {model.score(X_test, y_test):.3f}")
# R2: 0.927
```

## Tuning with GridSearchCV

ChebyshevRegressor is a standard sklearn estimator, so it works directly with
GridSearchCV, RandomizedSearchCV, cross_val_score, pipelines, etc.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {"complexity": [3, 5, 7, 9], "alpha": [0.01, 0.1, 1.0, 10.0]}
gs = GridSearchCV(ChebyshevRegressor(), param_grid, cv=5, scoring="r2")
gs.fit(X_train, y_train)
print(gs.best_params_)
# {'alpha': 0.1, 'complexity': 3}
print(f"Best CV R2: {gs.best_score_:.3f}")
# Best CV R2: 0.910
print(f"Test R2:    {gs.score(X_test, y_test):.3f}")
# Test R2:    0.932
```

## Inspecting fitted models

### Feature names and coefficient table

When fitted on a DataFrame, original column names are preserved in the
polynomial feature names. Array input falls back to `x0`, `x1`, etc.

```python
import pandas as pd

X_train_df = pd.DataFrame(X_train, columns=["speed", "temp", "pressure", "humidity", "voltage"])

model = ChebyshevRegressor(complexity=5, alpha=0.1).fit(X_train_df, y_train)

model.get_feature_names_out()[:6]
# array(['speed_T0', 'speed_T1', 'speed_T2', 'speed_T3', 'speed_T4',
#        'speed_T5'], dtype='<U11')

model.coef_table().head(6)
#        feature      coef input_feature
# 0     speed_T0 14.068879         speed
# 1  humidity_T1  4.899597      humidity
# 2     speed_T1  3.218066         speed
# 3      temp_T1  3.163345          temp
# 4   voltage_T1  2.479831       voltage
# 5  pressure_T2  2.413147      pressure
```

### Feature importances

Per-input-feature importances are computed by aggregating absolute coefficient
values across all polynomial terms for each original feature (excluding the
intercept term), normalised to sum to 1.

```python
for name, imp in zip(X_train_df.columns, model.feature_importances_):
    print(f"  {name:>10s}: {imp:.3f}")
#       speed: 0.245
#        temp: 0.244
#    pressure: 0.128
#    humidity: 0.249
#     voltage: 0.133
```

### Feature mapping

```python
mapping = model.get_feature_mapping()
mapping["temp"]
# ['temp_T1', 'temp_T2', 'temp_T3', 'temp_T4', 'temp_T5']
mapping["speed"]
# ['speed_T0', 'speed_T1', 'speed_T2', 'speed_T3', 'speed_T4', 'speed_T5']
```

Note: the first input feature retains T0 (the intercept/constant term); remaining
features start at T1 to avoid redundant constant columns in the design matrix.

## Model tree example

```python
from poly_basis_ml import ChebyshevModelTreeRegressor

X, y = make_friedman1(n_samples=2000, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = ChebyshevModelTreeRegressor(max_depth=3, complexity=3, alpha=0.1, min_samples_leaf=100)
model.fit(X_train, y_train)
print(f"R2:          {model.score(X_test, y_test):.3f}")
# R2:          0.951
print(f"Leaf models: {len(model.leaf_models_)}")
# Leaf models: 7
```

## Key features

- **ChebyshevExpander** --- standalone sklearn transformer for polynomial feature expansion
- **ChebyshevRegressor** --- convenience estimator wrapping Chebyshev expansion + Ridge
- **ChebyshevModelTreeRegressor** --- decision tree with Chebyshev polynomial leaf models
- **Bivariate interactions** --- optional product, contrast, and additive interaction features

## How it works

Features are mapped to [-1, 1] via MinMaxScaler, then expanded into Chebyshev
polynomial basis functions with proper intercept handling (one T0 term retained,
redundant constant columns stripped). The resulting design matrix is fitted with
Ridge regression. For the model tree variant, a decision tree first partitions
the data into regions, then each leaf fits a separate ChebyshevRegressor for
smooth local approximation.

## Main classes

### ChebyshevRegressor

| Parameter | Default | Description |
|-----------|---------|-------------|
| `complexity` | `5` | Chebyshev polynomial degree |
| `alpha` | `1.0` | Ridge regularisation strength |
| `clip_input` | `True` | Clip prediction-time inputs to training range |
| `include_interactions` | `False` | Add bivariate interaction features |

### ChebyshevModelTreeRegressor

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_depth` | `3` | Maximum depth of routing tree |
| `min_samples_leaf` | `200` | Minimum samples per leaf for polynomial fit |
| `complexity` | `2` | Chebyshev degree for leaf models |
| `alpha` | `10.0` | Ridge regularisation for leaf models |

## Citation

```bibtex
@article{gerber2026revisiting,
  title={Revisiting Chebyshev Polynomial and Anisotropic RBF Models for Tabular Regression},
  author={Gerber, Luciano and Lloyd, Huw},
  year={2026}
}
```

## Licence

MIT
