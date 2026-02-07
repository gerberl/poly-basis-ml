"""
poly_basis_ml - Chebyshev polynomial feature expansion and regression.
"""

from .expanders import ChebyshevExpander
from .regressor import ChebyshevRegressor
from .model_tree import ChebyshevModelTreeRegressor
from .interactions import INTERACTION_FUNCS, RECOMMENDED_INTERACTION_TYPES

__all__ = [
    'ChebyshevExpander',
    'ChebyshevRegressor',
    'ChebyshevModelTreeRegressor',
    'INTERACTION_FUNCS',
    'RECOMMENDED_INTERACTION_TYPES',
]

__version__ = '0.1.0'
