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

    Parameters
    ----------
    add_proc_number : bool (optional)
        Add processor identifier to multi-processor log messages. default True.
    """
    def __init__(self, add_proc_number=True):
        super().__init__()
        self._add_proc_number = add_proc_number

    def filter(self, record):
        if USING_MPI:
            if record.levelno == INFO:
                return MPIRANK == 0
            if self._add_proc_number:
                record.msg = "{}>\t".format(MPIRANK) + record.msg
        return True


class StatsFilter(logging.Filter):
    """This is a filter which filters based on the identifier "<stats>" at the
    beginning of a log message

    Parameters
    ----------
    filter_out : bool
        Whether to filter-out or filter-in stats messages
    """
    def __init__(self, filter_out):
        super().__init__()
        self._identifier = "<stats>"
        self._filter_out = filter_out

    def filter(self, record):
        if record.msg.startswith(self._identifier):
            if self._filter_out:
                return False
            record.msg = record.msg[len(self._identifier):]
            return True
        else:
            return self._filter_out
