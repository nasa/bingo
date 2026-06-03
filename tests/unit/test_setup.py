import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import setuptools


SETUP_PATH = Path(__file__).resolve().parents[2] / "setup.py"


@pytest.fixture
def setup_module(monkeypatch):
    monkeypatch.setattr(setuptools, "setup", lambda *args, **kwargs: None)

    spec = importlib.util.spec_from_file_location("bingo_setup", SETUP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_detects_mingw_shell_from_msystem(setup_module):
    assert setup_module._is_windows_gnu_toolchain(
        {"MSYSTEM": "MINGW64"}, platform="win32"
    )


def test_ninja_generator_args_use_distinct_cmake_flags(setup_module, monkeypatch):
    ninja_module = ModuleType("ninja")
    ninja_module.BIN_DIR = "/tmp/ninja-bin"
    monkeypatch.setitem(sys.modules, "ninja", ninja_module)

    assert setup_module._get_ninja_cmake_args() == [
        "-G",
        "Ninja",
        "-DCMAKE_JOB_POOLS:STRING=compile=1;link=1",
        "-DCMAKE_MAKE_PROGRAM:FILEPATH=/tmp/ninja-bin/ninja",
    ]


def test_windows_gnu_toolchain_falls_back_to_mingw_makefiles(
    setup_module, monkeypatch
):
    monkeypatch.setattr(setup_module, "_get_ninja_cmake_args", lambda: None)

    assert setup_module._get_single_config_generator_args(
        "", prefer_windows_gnu=True
    ) == ["-G", "MinGW Makefiles"]