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
  author={Gerber, Luciano and Lloyd, Chris},
  year={2026}
}
```

## Licence

MIT
