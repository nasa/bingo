/**
 * @file bind_cas_simplify.cpp
 * @brief pybind11 bindings for the CAS simplification pipeline.
 */

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "cppagraph/cas_simplify.h"

namespace py = pybind11;

void bind_cas_simplify(py::module_& m) {
    m.def("cas_simplify",
          [](const cppagraph::StackMatrix& raw_stack,
             const std::vector<double>& raw_constants,
             const std::vector<int>& raw_integers) {
              auto result = cppagraph::cas_simplify(
                  raw_stack, raw_constants, raw_integers);
              return py::make_tuple(
                  std::move(result.stack),
                  std::move(result.constants),
                  std::move(result.integers),
                  std::move(result.constant_mapping));
          },
          py::arg("raw_stack"),
          py::arg("raw_constants"),
          py::arg("raw_integers"),
          R"doc(
          Run the full CAS simplification pipeline.

          Parameters
          ----------
          raw_stack : numpy.ndarray
              Nx3 uint8 command array.
          raw_constants : list of float
              Constant values.
          raw_integers : list of int
              Integer values.

          Returns
          -------
          tuple
              (stack, constants, integers, constant_mapping).
          )doc");
}
