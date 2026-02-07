# Backward-compat shim: old module path for joblib files pickled before rename.
from .regressor import ChebyshevRegressor as PolyBasisRegressor  # noqa: F401
from ._vandermonde import _VandermondeTransform  # noqa: F401
from .interactions import _PolyInteractionTransform  # noqa: F401
