#!/usr/bin/env bash
# Build the cppagraph C++ extension for development.
#
# Usage:
#   ./.build_cppagraph.sh          # Release build + run C++ tests
#   ./.build_cppagraph.sh Debug    # Debug build + run C++ tests
#   ./.build_cppagraph.sh --no-test  # Build without running tests
#
# After a successful build the shared library is copied into the
# cppagraph package directory so that ``import bingo.expressions.agraph.cppagraph``
# picks it up immediately (no pip install needed).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/bingo/expressions/agraph/cppagraph"
BUILD_DIR="${SRC_DIR}/_build"

BUILD_TYPE="Release"
RUN_TESTS=true

for arg in "$@"; do
    case "$arg" in
        Debug|debug|RelWithDebInfo)
            BUILD_TYPE="$arg" ;;
        --no-test|--no-tests)
            RUN_TESTS=false ;;
    esac
done

# Detect cmake
if command -v cmake &>/dev/null; then
    CMAKE=cmake
elif [ -x "$(python3 -c 'import cmake, os; print(os.path.join(cmake.CMAKE_BIN_DIR, "cmake"))' 2>/dev/null)" ]; then
    CMAKE="$(python3 -c 'import cmake, os; print(os.path.join(cmake.CMAKE_BIN_DIR, "cmake"))')"
else
    echo "ERROR: cmake not found. Install via 'pip install cmake' or your package manager."
    exit 1
fi

# Detect ctest (same directory as cmake)
CTEST="$(dirname "$CMAKE")/ctest"
if [ ! -x "$CTEST" ]; then
    CTEST=ctest
fi

NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo "=== cppagraph build ==="
echo "  Source : ${SRC_DIR}"
echo "  Build  : ${BUILD_DIR}"
echo "  Type   : ${BUILD_TYPE}"
echo "  Jobs   : ${NPROC}"
echo ""

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

"${CMAKE}" .. \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCPPAGRAPH_BUILD_TESTS=ON

"${CMAKE}" --build . -j"${NPROC}"

if $RUN_TESTS; then
    echo ""
    echo "=== Running C++ tests ==="
    "${CTEST}" --output-on-failure
fi

# Copy the built .so into the package directory
echo ""
echo "=== Copying shared library ==="
SO_FILE=$(find "${BUILD_DIR}" -maxdepth 1 -name '_cppagraph*.so' -o -name '_cppagraph*.pyd' | head -1)
if [ -n "${SO_FILE}" ]; then
    cp "${SO_FILE}" "${SRC_DIR}/"
    echo "  Copied: $(basename "${SO_FILE}") → ${SRC_DIR}/"
else
    echo "  WARNING: No .so/.pyd found in ${BUILD_DIR}"
fi

echo ""
echo "=== Done ==="
