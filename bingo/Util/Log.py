"""
This Logging module is just a simplified interface to the python built-in
logging library.  Its sets up default logging options which are typical of most
bingo runs.
"""
import logging
import warnings

INFO = 25
DETAILED_INFO = 20

try:
    import mpi4py
    MPISIZE = mpi4py.MPI.COMM_WORLD.Get_size()
    MPIRANK = mpi4py.MPI.COMM_WORLD.Get_rank()
    USING_MPI = MPISIZE > 1
except (ImportError, AttributeError):
    USING_MPI = False


def configure_logging(verbosity="standard", module=False, timestamp=False):
    level = _get_log_level_from_verbosity(verbosity)
    format_string = _get_format_string(module, timestamp)
    logging.basicConfig(level=level, format=format_string)
    root_logger = logging.getLogger()
    root_logger.handlers[0].addFilter(MpiFilter())


def _get_log_level_from_verbosity(verbosity):
    verbosity_map = {"quiet": logging.WARNING,
                     "standard": INFO,
                     "detailed": DETAILED_INFO,
                     "debug": logging.DEBUG}
    if isinstance(verbosity, str):
        return verbosity_map[verbosity]
    elif isinstance(verbosity, int):
        return verbosity
    else:
        warnings.warn("Unrecognized verbosity level provided. "
                      "Using standard verbosity.")
        return INFO


def _get_format_string(module, timestamp):
    format_string = "%(message)s"
    if module:
        format_string = "%(module)s\t" + format_string
    if timestamp:
        format_string = "%(asctime)s\t" + format_string
    return format_string


class MpiFilter(logging.Filter):
    """
    This is a filter which filters out messages from auxiliary processes at the
    INFO level
    """
    def filter(self, record):
        if USING_MPI:
            if record.levelno == INFO:
                return MPIRANK == 0
            record.msg = "{}>\t".format(MPIRANK) + record.msg
        return True
