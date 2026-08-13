"""Tests for the serial package installation contract."""

import importlib
import importlib.abc
import sys

import pytest


class _BlockMpi4py(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, _path=None, _target=None):
        """Report mpi4py as unavailable while allowing all other imports."""
        if fullname == "mpi4py":
            raise ModuleNotFoundError("No module named 'mpi4py'", name="mpi4py")


def test_serial_imports_do_not_require_mpi4py():
    """Serial package imports remain available without mpi4py."""
    bingo = importlib.import_module("bingo")
    log = importlib.import_module("bingo.util.log")

    assert bingo is not None
    assert not log.USING_MPI


def test_parallel_archipelago_missing_mpi_extra_is_actionable(monkeypatch):
    """The parallel module directs users to the MPI optional dependency."""
    monkeypatch.setattr(sys, "meta_path", [_BlockMpi4py(), *sys.meta_path])
    for module_name in tuple(sys.modules):
        if module_name == "mpi4py" or module_name.startswith("mpi4py."):
            monkeypatch.delitem(sys.modules, module_name)
    monkeypatch.delitem(
        sys.modules,
        "bingo.evolutionary_optimizers.parallel_archipelago",
        raising=False,
    )

    with pytest.raises(ImportError, match=r"bingo-nasa\[MPI\]") as error:
        importlib.import_module("bingo.evolutionary_optimizers.parallel_archipelago")

    assert isinstance(error.value.__cause__, ModuleNotFoundError)
