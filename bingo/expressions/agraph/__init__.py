"""Acyclic graph expression for symbolic regression.

If the C++ accelerated backend (cppagraph) is available, its
``AGraphExpression`` is used automatically.  Otherwise the pure-Python
implementation from pyagraph is loaded as a fallback.

Use :func:`set_backend` to override the automatic selection::

    from bingo.expressions.agraph import set_backend
    set_backend("python")   # force pure-Python backend
    set_backend("cpp")      # force C++ backend (raises ImportError if unavailable)
    set_backend("auto")     # restore automatic selection

Backend switches are visible through module attribute access or
:func:`get_expression_class`::

    import bingo.expressions.agraph as agraph

    set_backend("python")
    expr = agraph.AGraphExpression("x0 + x1")

A directly imported class name is bound eagerly by Python and will not
track later :func:`set_backend` calls::

    from bingo.expressions.agraph import AGraphExpression, set_backend

    set_backend("python")
    expr = AGraphExpression("x0 + x1")
    # Uses whichever class AGraphExpression referred to at import time.
"""

from .pyagraph import AGraphExpression as _PyAGraphExpression

try:
    from .cppagraph import AGraphExpression as _CppAGraphExpression

    _cpp_available = True
except ImportError:
    _CppAGraphExpression = None
    _cpp_available = False

_backend = "auto"
AGraphExpression = _CppAGraphExpression if _cpp_available else _PyAGraphExpression


def _sync_parent_reexports():
    """Keep the top-level expressions re-export aligned with backend switches."""
    import sys

    expressions_pkg = sys.modules.get("bingo.expressions")
    if expressions_pkg is not None:
        expressions_pkg.AGraphExpression = AGraphExpression


_sync_parent_reexports()


def set_backend(backend):
    """Select which AGraphExpression backend to use.

    Parameters
    ----------
    backend : {"auto", "python", "cpp"}
        ``"auto"`` uses C++ if available, otherwise Python.
        ``"python"`` forces the pure-Python implementation.
        ``"cpp"`` forces the C++ implementation.

    Raises
    ------
    ImportError
        If ``"cpp"`` is requested but cppagraph is not installed.
    ValueError
        If *backend* is not one of the recognised values.

    Notes
    -----
    Access the class through ``bingo.expressions.agraph.AGraphExpression`` or
    :func:`get_expression_class` after switching backends. A local name created
    with ``from bingo.expressions.agraph import AGraphExpression`` will keep the
    class object that was imported originally.
    """
    global AGraphExpression, _backend  # noqa: PLW0603
    if backend == "auto":
        AGraphExpression = (
            _CppAGraphExpression if _cpp_available else _PyAGraphExpression
        )
    elif backend == "python":
        AGraphExpression = _PyAGraphExpression
    elif backend == "cpp":
        if not _cpp_available:
            raise ImportError(
                "cppagraph C++ backend is not available. "
                "Build it with .build_cppagraph.sh or use "
                "set_backend('python')."
            )
        AGraphExpression = _CppAGraphExpression
    else:
        raise ValueError(
            f"Unknown backend {backend!r}. " f"Choose from 'auto', 'python', or 'cpp'."
        )
    _backend = backend
    _sync_parent_reexports()


def get_backend():
    """Return the name of the active backend.

    Returns
    -------
    str
        ``"auto"``, ``"python"``, or ``"cpp"``.
    """
    return _backend


def get_expression_class():
    """Return the currently active ``AGraphExpression`` class.

    This always reflects the latest :func:`set_backend` call and is
    intended for call-sites that need deferred resolution (e.g. the
    generator).

    Returns
    -------
    type
    """
    return AGraphExpression


from .component_generator import ComponentGenerator
from .generator import AGraphGenerator
from .utils import pad_agraph_expression
from .crossover import AGraphCrossover
from .mutation import AGraphMutation

__all__ = [
    "AGraphExpression",
    "ComponentGenerator",
    "AGraphGenerator",
    "pad_agraph_expression",
    "AGraphCrossover",
    "AGraphMutation",
    "set_backend",
    "get_backend",
    "get_expression_class",
]
