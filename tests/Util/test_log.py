# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import logging

from bingo.Util import Log


@pytest.mark.parametrize("verbosity, expected_level",
                         [("debug", 10),
                          ("detailed", Log.DETAILED_INFO),
                          ("standard", Log.INFO),
                          ("quiet", 30),
                          (31, 31),
                          (0.5, 25)])
def test_configure_logging_verbosity(verbosity, expected_level, mocker):
    mocker.patch('logging.Logger.setLevel')
    Log.configure_logging(verbosity)
    logging.Logger.setLevel.assert_called_with(expected_level)


@pytest.mark.parametrize("module", [True, False])
@pytest.mark.parametrize("timestamp", [True, False])
def test_configure_logging_formatting(module, timestamp, mocker):
    mocker.patch('logging.StreamHandler.setFormatter')
    Log.configure_logging(module=module, timestamp=timestamp)
    positional_args, _ = logging.StreamHandler.setFormatter.call_args
    formatter = positional_args[0]
    assert ("module" in formatter._fmt) == module
    assert ("asctime" in formatter._fmt) == timestamp


def test_configure_logging_makes_console_handler(mocker):
    mocker.patch('logging.Logger.addHandler')
    Log.configure_logging()
    positional_args, _ = logging.Logger.addHandler.call_args
    assert isinstance(positional_args[0], logging.StreamHandler)


def test_console_handler_gets_two_filters(mocker):
    mocker.patch('logging.StreamHandler.addFilter')
    Log.configure_logging()
    calls = logging.StreamHandler.addFilter.call_args_list
    filters = [positional_args[0] for positional_args, _ in calls]
    assert isinstance(filters[0], Log.StatsFilter)
    assert isinstance(filters[1], Log.MpiFilter)


def test_configure_logging_makes_stats_file_handler(mocker):
    mocker.patch('logging.Logger.addHandler')
    Log.configure_logging(stats_file="test.log")
    positional_args, _ = logging.Logger.addHandler.call_args
    assert isinstance(positional_args[0], logging.FileHandler)


def test_stats_file_handler_gets_two_filters(mocker):
    mocker.patch('logging.FileHandler.addFilter')
    Log.configure_logging(stats_file="test.log")
    calls = logging.FileHandler.addFilter.call_args_list
    filters = [positional_args[0] for positional_args, _ in calls]
    assert isinstance(filters[0], Log.StatsFilter)
    assert isinstance(filters[1], Log.MpiFilter)


@pytest.mark.parametrize("level, mpi_on, mpi_rank, expected_filter",
                         [(Log.INFO, True, 0, True),
                          (Log.INFO, True, 1, False),
                          (Log.INFO, False, None, True),
                          (Log.DETAILED_INFO, True, 0, True),
                          (Log.DETAILED_INFO, True, 1, True),
                          (Log.DETAILED_INFO, False, None, True)])
def test_mpi_filtering(level, mpi_on, mpi_rank, expected_filter, mocker):
    Log.USING_MPI = mpi_on
    Log.MPIRANK = mpi_rank
    record = mocker.Mock()
    record.levelno = level
    record.msg = ""

    mpi_filter = Log.MpiFilter()
    assert mpi_filter.filter(record) == expected_filter


@pytest.mark.parametrize("add_proc_num", [True, False])
@pytest.mark.parametrize("mpi_on", [True, False])
def test_mpi_filter_adds_proc_num(add_proc_num, mpi_on, mocker):
    Log.USING_MPI = mpi_on
    Log.MPIRANK = 0
    record = mocker.Mock()
    record.levelno = Log.DETAILED_INFO
    record.msg = ""

    mpi_filter = Log.MpiFilter(add_proc_num)
    proc_num_expected = add_proc_num and mpi_on
    _ = mpi_filter.filter(record)
    assert ("0>" in record.msg) == proc_num_expected


@pytest.mark.parametrize("filter_out", [True, False])
@pytest.mark.parametrize("stats_extra", [True, False, None])
def test_mpi_filter_adds_proc_num(filter_out, stats_extra, mocker):
    record = mocker.Mock()
    if stats_extra is not None:
        record.stats = stats_extra

    stats_filter = Log.StatsFilter(filter_out)
    if stats_extra is None:
        expected_filter = filter_out
    else:
        expected_filter = filter_out != stats_extra
    assert stats_filter.filter(record) == expected_filter



