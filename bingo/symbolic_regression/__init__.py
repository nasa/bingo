"""
Import the core names of bingo symbolic_regression library

Programs that want to build bingo symbolic regression apps
without having to import specific modules can import this.
"""

from .agraph.component_generator import *
from .agraph.crossover import *
from .agraph.generator import *
from .agraph.mutation import *
from .atomic_potential_regression import *

# Try to load in C++ cpython extensions
# TODO: consider this init file but remove imports for python
# that have C++ bindings
try:
    from bingocpp.symbolic_regression import *
    ISCPP = True
except ImportError as e:
    from .agraph.agraph import *
    from .equation import *
    from .explicit_regression import *
    from .implicit_regression import *
    ISCPP = False
    print("Could not load C++ modules")