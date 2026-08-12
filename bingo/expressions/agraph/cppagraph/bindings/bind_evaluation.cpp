/**
 * @file bind_evaluation.cpp
 * @brief pybind11 bindings for evaluate(), evaluate_with_derivative(),
 *        and the CachedEvaluator class.
 *
 * Exposes the same function signatures as
 * bingo.expressions.agraph.pyagraph.evaluation so that the Python
 * facade can be a transparent drop-in replacement.
 */

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "cppagraph/evaluation.h"
#include "cppagraph/cached_evaluation.h"

namespace py = pybind11;
using namespace cppagraph;

void bind_evaluation(py::module_& m) {

    // ---- Free functions ----

    m.def("evaluate",
        [](py::array_t<uint8_t, py::array::c_style | py::array::forcecast> stack_arr,
           const RowMatrixXd& x,
           const std::vector<double>& constants,
           const std::vector<int>& integers) -> RowMatrixXd
        {
            auto buf = stack_arr.request();
            if (buf.ndim != 2 || buf.shape[1] != 3)
                throw std::invalid_argument("stack must be Nx3 uint8 array");
            Eigen::Map<const StackMatrix> stack(
                static_cast<const uint8_t*>(buf.ptr),
                buf.shape[0], 3);
            return evaluate(stack, x, constants, integers);
        },
        py::arg("stack"), py::arg("x"),
        py::arg("constants"), py::arg("integers"),
        "Evaluate an equation represented by a command stack.\n\n"
        "Parameters\n"
        "----------\n"
        "stack : (N, 3) uint8 array\n"
        "x : (M, D) float64 array\n"
        "constants : tuple of float\n"
        "integers : tuple of int\n\n"
        "Returns\n"
        "-------\n"
        "(M, 1) float64 array");

    m.def("evaluate_with_derivative",
        [](py::array_t<uint8_t, py::array::c_style | py::array::forcecast> stack_arr,
           const RowMatrixXd& x,
           const std::vector<double>& constants,
           const std::vector<int>& integers,
           bool wrt_x) -> std::pair<RowMatrixXd, RowMatrixXd>
        {
            auto buf = stack_arr.request();
            if (buf.ndim != 2 || buf.shape[1] != 3)
                throw std::invalid_argument("stack must be Nx3 uint8 array");
            Eigen::Map<const StackMatrix> stack(
                static_cast<const uint8_t*>(buf.ptr),
                buf.shape[0], 3);
            return evaluate_with_derivative(stack, x, constants, integers, wrt_x);
        },
        py::arg("stack"), py::arg("x"),
        py::arg("constants"), py::arg("integers"),
        py::arg("wrt_param_x_or_c"),
        "Evaluate and compute derivative via reverse-mode autodiff.\n\n"
        "Parameters\n"
        "----------\n"
        "stack : (N, 3) uint8 array\n"
        "x : (M, D) float64 array\n"
        "constants : tuple of float\n"
        "integers : tuple of int\n"
        "wrt_param_x_or_c : bool\n"
        "    True for df/dx, False for df/dc.\n\n"
        "Returns\n"
        "-------\n"
         "tuple of ((M, 1), (M, D) or (M, L)) float64 arrays");

    m.def("evaluate_with_const_hessian",
        [](py::array_t<uint8_t, py::array::c_style | py::array::forcecast> stack_arr,
           const RowMatrixXd& x, const std::vector<double>& constants,
           const std::vector<int>& integers) {
            auto buf = stack_arr.request();
            if (buf.ndim != 2 || buf.shape[1] != 3)
                throw std::invalid_argument("stack must be Nx3 uint8 array");
            Eigen::Map<const StackMatrix> stack(
                static_cast<const uint8_t*>(buf.ptr), buf.shape[0], 3);
            ConstHessianResult result;
            try {
                result = evaluate_with_const_hessian(stack, x, constants, integers);
            } catch (const std::domain_error& err) {
                PyErr_SetString(PyExc_ZeroDivisionError, err.what());
                throw py::error_already_set();
            }
            const auto l = static_cast<py::ssize_t>(constants.size());
            auto hessian = py::array_t<double>({x.rows(), l, l});
            auto hessian_view = hessian.mutable_unchecked<3>();
            for (Eigen::Index row = 0; row < x.rows(); ++row)
                for (py::ssize_t j = 0; j < l; ++j)
                    for (py::ssize_t k = 0; k < l; ++k)
                        hessian_view(row, j, k) = result.hessian(row, j * l + k);
            return py::make_tuple(result.value, result.gradient, hessian);
        },
        py::arg("stack"), py::arg("x"), py::arg("constants"), py::arg("integers"));

    // ---- CachedEvaluator class ----

    py::class_<CachedEvaluator>(m, "CachedEvaluator",
        "Evaluation context with caching for repeated constant-only changes.\n\n"
        "Create at the start of fit(), discard when fitting is complete.")

        .def(py::init(
            [](py::array_t<uint8_t, py::array::c_style | py::array::forcecast> stack_arr,
               const RowMatrixXd& x,
               const std::vector<int>& integers) {
                auto buf = stack_arr.request();
                if (buf.ndim != 2 || buf.shape[1] != 3)
                    throw std::invalid_argument("stack must be Nx3 uint8 array");
                StackMatrix stack(buf.shape[0], 3);
                std::memcpy(stack.data(), buf.ptr,
                            buf.shape[0] * 3 * sizeof(uint8_t));
                return CachedEvaluator(std::move(stack), x,
                                        integers);
            }),
            py::arg("stack"), py::arg("x"), py::arg("integers"),
            "Construct a CachedEvaluator.\n\n"
            "Parameters\n----------\n"
            "stack : (N, 3) uint8 array\n"
            "x : (M, D) float64 array\n"
            "integers : tuple of int")

        .def("forward_eval",
            &CachedEvaluator::forward_eval,
            py::arg("constants"),
            "Evaluate f(x) with the given constants.")

        .def("forward_eval_with_const_derivative",
            &CachedEvaluator::forward_eval_with_const_derivative,
            py::arg("constants"),
            "Evaluate f(x) and df/dc with the given constants.")
    ;
}
