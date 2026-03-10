cmake_minimum_required(VERSION 3.16)
project(cppagraph LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

# pybind11 — prefer installed, fall back to FetchContent
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Eigen3 — prefer installed, fall back to FetchContent
find_package(Eigen3 3.4 QUIET)
if(NOT Eigen3_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
        GIT_SHALLOW    TRUE
    )
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(eigen)
    # Provide the target alias that find_package would create
    if(NOT TARGET Eigen3::Eigen)
        add_library(Eigen3::Eigen ALIAS eigen)
    endif()
endif()

# ---------------------------------------------------------------------------
# Core static library (shared between pybind module and C++ tests)
# ---------------------------------------------------------------------------

add_library(cppagraph_core STATIC
    src/operators.cpp
    src/data_container.cpp
    src/operator_eval.cpp
    src/evaluation.cpp
    src/cached_evaluation.cpp
    src/simplification.cpp
    src/cas_expression.cpp
    src/automatic_simplification.cpp
    src/interpreter.cpp
    src/constant_folding.cpp
    src/optional_modifications.cpp
    src/cas_simplify.cpp
    src/expression.cpp
)
target_include_directories(cppagraph_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(cppagraph_core PUBLIC Eigen3::Eigen)

# ---------------------------------------------------------------------------
# pybind11 Python extension module  (_cppagraph)
# ---------------------------------------------------------------------------

pybind11_add_module(_cppagraph
    bindings/module.cpp
    bindings/bind_operators.cpp
    bindings/bind_data_container.cpp
    bindings/bind_evaluation.cpp
    bindings/bind_simplification.cpp
    bindings/bind_cas_simplify.cpp
    bindings/bind_expression.cpp
)
target_link_libraries(_cppagraph PRIVATE cppagraph_core)

# ---------------------------------------------------------------------------
# C++ unit tests (Google Test)
# ---------------------------------------------------------------------------

option(CPPAGRAPH_BUILD_TESTS "Build C++ unit tests for cppagraph" ON)
if(CPPAGRAPH_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.15.2
        GIT_SHALLOW    TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    add_executable(cppagraph_tests
        tests/test_operators.cpp
        tests/test_data_container.cpp
        tests/test_evaluation.cpp
        tests/test_simplification.cpp
        tests/test_cas_simplify.cpp
        tests/test_expression.cpp
    )
    target_link_libraries(cppagraph_tests PRIVATE
        cppagraph_core
        GTest::gtest_main
    )
    include(GoogleTest)
    gtest_discover_tests(cppagraph_tests)
endif()
