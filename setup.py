import os
import re
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# from distutils.version import LooseVersion

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


def _is_windows_gnu_toolchain(env=None, platform=None):
    env = os.environ if env is None else env
    platform = sys.platform if platform is None else platform

    if not platform.startswith("win"):
        return False

    compiler_vars = " ".join(
        value for value in (env.get("CC", ""), env.get("CXX", "")) if value
    ).lower()
    msystem = env.get("MSYSTEM", "")

    return (
        msystem.startswith("MINGW")
        or bool(env.get("MINGW_PREFIX"))
        or bool(env.get("MINGW_CHOST"))
        or any(token in compiler_vars for token in ("mingw", "gcc", "g++", "clang"))
    )


def _get_ninja_cmake_args():
    try:
        import ninja
    except ImportError:
        return None

    ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
    return [
        "-G",
        "Ninja",
        "-DCMAKE_JOB_POOLS:STRING=compile=1;link=1",
        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
    ]


def _get_single_config_generator_args(cmake_generator, prefer_windows_gnu=False):
    if cmake_generator and cmake_generator != "Ninja":
        return []

    ninja_args = _get_ninja_cmake_args()
    if ninja_args is not None:
        return ninja_args

    if prefer_windows_gnu and not cmake_generator:
        return ["-G", "MinGW Makefiles"]

    return []


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        # Using this requires trailing slash for auto-detection & inclusion of
        # auxiliary "native" libs

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
        ]
        build_args = ["--target=_cppagraph"]
        # Skip C++ unit tests during pip install.
        cmake_args += ["-DCPPAGRAPH_BUILD_TESTS=OFF"]
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # In this example, we pass in the version to C++. You might not need to.
        cmake_args += [f"-DEXAMPLE_VERSION_INFO={self.distribution.get_version()}"]

        windows_gnu_toolchain = _is_windows_gnu_toolchain()

        if self.compiler.compiler_type != "msvc" or windows_gnu_toolchain:
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            cmake_args += _get_single_config_generator_args(
                cmake_generator, prefer_windows_gnu=windows_gnu_toolchain
            )

        else:
            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]
                build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        # # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # # across all generators.
        # if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
        #     # self.parallel is a Python 3 only way to set parallel jobs by hand
        #     # using -j in the build_ext call, not supported by pip or PyPA-build.
        #     if hasattr(self, "parallel") and self.parallel:
        #         # CMake 3.12+ only.
        #         build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", *build_args], cwd=build_temp, check=True
        )


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    packages=[
        "bingo",
        "bingo.chromosomes",
        "bingo.evaluation",
        "bingo.evolutionary_algorithms",
        "bingo.evolutionary_optimizers",
        "bingo.expressions",
        "bingo.expressions.agraph",
        "bingo.expressions.agraph.pyagraph",
        "bingo.expressions.agraph.pyagraph.evaluation",
        "bingo.expressions.agraph.pyagraph.simplification",
        "bingo.expressions.agraph.cppagraph",
        "bingo.local_optimizers",
        "bingo.selection",
        "bingo.stats",
        "bingo.symbolic_regression",
        "bingo.util",
        "bingo.variation",
    ],
    # add extension modules
    ext_modules=[
        CMakeExtension(
            "bingo.expressions.agraph.cppagraph._cppagraph",
            "bingo/expressions/agraph/cppagraph",
        ),
    ],
    # add custom build_ext command
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
