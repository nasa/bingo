/**
 * @file bind_simplification.cpp
 * @brief pybind11 bindings for stack reduction functions.
 *
 * Exposes get_utilized_commands() and reduce() to Python.
 */

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "cppagraph/simplification.h"

namespace py = pybind11;

void bind_simplification(py::module_& m) {
    using namespace cppagraph;

    m.def(
        "get_utilized_commands",
        [](py::array_t<uint8_t, py::array::c_style | py::array::forcecast> stack_arr) {
            auto buf = stack_arr.request();
            if (buf.ndim != 2 || buf.shape[1] != 3)
                throw std::runtime_error(
                    "get_utilized_commands: stack must be Nx3 uint8 array");
            Eigen::Index n = buf.shape[0];
            Eigen::Map<const StackMatrix> stack(
                static_cast<const uint8_t*>(buf.ptr), n, 3);
            auto util = get_utilized_commands(stack);

            // Return as a Python list of bools (matching pyagraph's bytearray)
            py::list result(n);
            for (Eigen::Index i = 0; i < n; ++i)
                result[i] = py::bool_(util[i]);
            return result;
        },
        py::arg("stack"),
        R"pbdoc(
        Find which commands are utilized by the final output.

        Parameters
        ----------
        stack : numpy.ndarray
            Nx3 uint8 command array.

        Returns
        -------
        list of bool
            True for each utilized row.
        )pbdoc"
    );

    m.def(
        "reduce",
        [](py::array_t<uint8_t, py::array::c_style | py::array::forcecast> stack_arr,
           const std::vector<double>& raw_constants,
           const std::vector<int>& raw_integers) {
            auto buf = stack_arr.request();
            if (buf.ndim != 2 || buf.shape[1] != 3)
                throw std::runtime_error(
                    "reduce: stack must be Nx3 uint8 array");
            Eigen::Index n = buf.shape[0];
            Eigen::Map<const StackMatrix> stack(
                static_cast<const uint8_t*>(buf.ptr), n, 3);

            auto result = reduce(stack, raw_constants, raw_integers);

            // Convert StackMatrix → numpy (N×3 uint8, C-contiguous copy)
            Eigen::Index rows = result.stack.rows();
            py::array_t<uint8_t> py_stack({rows, Eigen::Index(3)});
            auto py_buf = py_stack.mutable_unchecked<2>();
            for (Eigen::Index i = 0; i < rows; ++i)
                for (Eigen::Index j = 0; j < 3; ++j)
                    py_buf(i, j) = result.stack(i, j);

            // Return as a tuple matching the Python API:
            // (command_array, constants_tuple, integers_tuple, constant_mapping_tuple)
            py::tuple py_consts(result.constants.size());
            for (size_t i = 0; i < result.constants.size(); ++i)
                py_consts[i] = py::float_(result.constants[i]);

            py::tuple py_ints(result.integers.size());
            for (size_t i = 0; i < result.integers.size(); ++i)
                py_ints[i] = py::int_(result.integers[i]);

            py::tuple py_mapping(result.constant_mapping.size());
            for (size_t i = 0; i < result.constant_mapping.size(); ++i)
                py_mapping[i] = py::int_(result.constant_mapping[i]);

            return py::make_tuple(py_stack, py_consts, py_ints, py_mapping);
        },
        py::arg("raw_stack"),
        py::arg("raw_constants"),
        py::arg("raw_integers"),
        R"pbdoc(
        Reduce the raw stack with dead-code elimination and terminal renumbering.

        Parameters
        ----------
        raw_stack : numpy.ndarray
            Nx3 uint8 raw command array.
        raw_constants : list of float
            Constant values indexed by CONSTANT node params.
        raw_integers : list of int
            Integer values indexed by INTEGER node params.

        Returns
        -------
        tuple
            (command_array, constants, integers, constant_mapping).
        )pbdoc"
    );
}
