"""Acyclic graph expression for symbolic regression.

If the C++ accelerated backend (cppagraph) is available, its
``AGraphExpression`` is used automatically.  Otherwise the pure-Python
implementation from pyagraph is loaded as a fallback.
"""

try:
    from .cppagraph import AGraphExpression  # C++ backend
except ImportError:
    from .pyagraph import AGraphExpression  # Python fallback

from .component_generator import ComponentGenerator
from .generator import AGraphGenerator
from .crossover import AGraphCrossover
from .mutation import AGraphMutation

__all__ = [
    "AGraphExpression",
    "ComponentGenerator",
    "AGraphGenerator",
    "AGraphCrossover",
    "AGraphMutation",
]
