/**
 * @file module.cpp
 * @brief pybind11 module entry point for the cppagraph C++ extension.
 *
 * Sub-modules are registered by helper functions defined in their
 * respective bind_*.cpp translation units.
 */

#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations — implemented in bind_*.cpp files.
void bind_operators(py::module_& m);
void bind_data_container(py::module_& m);
void bind_evaluation(py::module_& m);
void bind_simplification(py::module_& m);
void bind_cas_simplify(py::module_& m);
void bind_expression(py::module_& m);

PYBIND11_MODULE(_cppagraph, m) {
    m.doc() = "C++17 accelerator for the AGraph expression engine";

    bind_operators(m);
    bind_data_container(m);
    bind_evaluation(m);
    bind_simplification(m);
    bind_cas_simplify(m);
    bind_expression(m);
}
