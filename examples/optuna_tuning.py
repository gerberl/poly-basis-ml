"""
Optuna hyperparameter tuning for ChebyshevRegressor and
ChebyshevModelTreeRegressor.

Search spaces adapted from the poly_erbf_benchmark project.

Usage:
    python examples/optuna_tuning.py
"""

import numpy as np
import optuna
from sklearn.datasets import make_friedman1
from sklearn.model_selection import cross_val_score, KFold

from poly_basis_ml import ChebyshevRegressor, ChebyshevModelTreeRegressor


# ---------------------------------------------------------------------------
# ChebyshevRegressor (pure polynomial expansion + Ridge)
# ---------------------------------------------------------------------------

def chebypoly_objective(trial: optuna.Trial, X, y):
    """Optuna objective: 5-fold CV R² for ChebyshevRegressor."""
    complexity = trial.suggest_int("complexity", 1, 14)
    alpha = trial.suggest_float("alpha", 1e-3, 1e3, log=True)
    include_interactions = trial.suggest_categorical("include_interactions", [False, True])

    if include_interactions:
        max_interaction_complexity = trial.suggest_categorical(
            "max_interaction_complexity", [1, 2]
        )
        expand_interactions = max_interaction_complexity > 1
    else:
        max_interaction_complexity = 1
        expand_interactions = False

    model = ChebyshevRegressor(
        complexity=complexity,
        alpha=alpha,
        clip_input=True,
        include_interactions=include_interactions,
        interaction_types=["product"] if include_interactions else None,
        expand_interactions=expand_interactions,
        max_interaction_complexity=max_interaction_complexity,
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    return scores.mean()


# ---------------------------------------------------------------------------
# ChebyshevModelTreeRegressor (decision tree with polynomial leaves)
# ---------------------------------------------------------------------------

def chebytree_objective(trial: optuna.Trial, X, y):
    """Optuna objective: 5-fold CV R² for ChebyshevModelTreeRegressor."""
    model = ChebyshevModelTreeRegressor(
        complexity=trial.suggest_int("complexity", 1, 6),
        alpha=trial.suggest_float("alpha", 1e-3, 1e3, log=True),
        max_depth=trial.suggest_int("max_depth", 1, 12),
        min_samples_leaf=trial.suggest_float("min_samples_leaf", 0.01, 0.1),
        random_state=42,
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    return scores.mean()


if __name__ == "__main__":
    X, y = make_friedman1(n_samples=500, n_features=5, noise=0.5, random_state=42)

    # --- ChebyshevRegressor ---
    print("=" * 60)
    print("ChebyshevRegressor")
    print("=" * 60)
    study_poly = optuna.create_study(direction="maximize")
    study_poly.optimize(lambda t: chebypoly_objective(t, X, y), n_trials=30)
    print(f"Best R²: {study_poly.best_value:.4f}")
    print(f"Best params: {study_poly.best_params}")

    # --- ChebyshevModelTreeRegressor ---
    print("\n" + "=" * 60)
    print("ChebyshevModelTreeRegressor")
    print("=" * 60)
    study_tree = optuna.create_study(direction="maximize")
    study_tree.optimize(lambda t: chebytree_objective(t, X, y), n_trials=30)
    print(f"Best R²: {study_tree.best_value:.4f}")
    print(f"Best params: {study_tree.best_params}")
