"""
This Logging module is just a simplified interface to the python built-in
logging library.  Its sets up default logging options which are typical of most
bingo runs.
"""
import logging

INFO = 25
DETAILED_INFO = 20

try:
    import mpi4py
    MPIRANK = mpi4py.MPI.COMM_WORLD.Get_rank()
except ImportError:
    mpi4py = None


def set_default_logging():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root_logger = logging.getLogger()
    root_logger.handlers[0].addFilter(MpiFilter())


class MpiFilter(logging.Filter):
    """
    This is a filter which filters out messages from auxiliary processes at the
    INFO level
    """
    def filter(self, record):
        if mpi4py is not None:
            if record.levelno == INFO:
                return MPIRANK == 0
            record.msg = "{}>\t".format(MPIRANK) + record.msg
        return True
