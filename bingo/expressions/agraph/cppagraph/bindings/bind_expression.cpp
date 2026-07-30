/**
 * @file bind_expression.cpp
 * @brief pybind11 bindings for AGraphExpression.
 *
 * Provides the same Python-level API as
 * bingo.expressions.agraph.pyagraph.expression.AGraphExpression,
 * including equation-string construction (via pyagraph parsing),
 * formatting properties (via pyagraph formatting), pickle, and
 * deepcopy.
 */

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "cppagraph/expression.h"

#include <cstring>

namespace py = pybind11;
using namespace cppagraph;

// ---- Helpers --------------------------------------------------- //

namespace {

/// Convert a StackMatrix to a read-only numpy uint8 array (copy).
py::array_t<uint8_t> stack_to_numpy(const StackMatrix& cmd) {
    auto rows = static_cast<py::ssize_t>(cmd.rows());
    auto cols = static_cast<py::ssize_t>(cmd.cols());
    auto result = py::array_t<uint8_t>({rows, cols});
    if (rows > 0) {
        std::memcpy(result.mutable_data(), cmd.data(),
                     static_cast<size_t>(rows * cols));
    }
    // Make read-only
    py::detail::array_proxy(result.ptr())->flags &=
        ~py::detail::npy_api::NPY_ARRAY_WRITEABLE_;
    return result;
}

/// Convert a numpy uint8 array to a StackMatrix (copy).
StackMatrix numpy_to_stack(
        py::array_t<uint8_t, py::array::c_style |
                              py::array::forcecast> arr) {
    auto buf = arr.request();
    if (buf.ndim != 2 || buf.shape[1] != 3)
        throw std::invalid_argument("Expected Nx3 uint8 array");
    Eigen::Index rows = buf.shape[0];
    StackMatrix cmd(rows, 3);
    if (rows > 0) {
        std::memcpy(cmd.data(), buf.ptr,
                     static_cast<size_t>(rows * 3));
    }
    return cmd;
}

/// std::vector<double> → Python tuple of float
py::tuple vec_to_float_tuple(const std::vector<double>& v) {
    py::tuple t(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        t[i] = py::float_(v[i]);
    return t;
}

/// std::vector<int> → Python tuple of int
py::tuple vec_to_int_tuple(const std::vector<int>& v) {
    py::tuple t(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        t[i] = py::int_(v[i]);
    return t;
}

/// Python iterable → std::vector<double>
std::vector<double> iterable_to_double_vec(py::object obj) {
    std::vector<double> v;
    for (auto item : obj) v.push_back(item.cast<double>());
    return v;
}

/// Python iterable → std::vector<int>
std::vector<int> iterable_to_int_vec(py::object obj) {
    std::vector<int> v;
    for (auto item : obj) v.push_back(item.cast<int>());
    return v;
}

}  // anonymous namespace

// ---- Binding --------------------------------------------------- //

void bind_expression(py::module_& m) {

    // ---- Scoring metrics as module-level functions ---- //
    m.def("mean_absolute_error", &mean_absolute_error,
          py::arg("residuals"));
    m.def("mean_squared_error", &mean_squared_error,
          py::arg("residuals"));
    m.def("root_mean_squared_error", &root_mean_squared_error,
          py::arg("residuals"));
    m.def("bic_score", &bic_score,
          py::arg("residuals"), py::arg("n_constants"));
    m.def("laplace_nmll_score", &laplace_nmll_score,
          py::arg("residuals"), py::arg("n_constants"));

    // ---- AGraphExpression class ---- //
    py::class_<AGraphExpression>(m, "AGraphExpression")

        // ---- Construction ----
        .def(py::init(
            [](py::object equation,
               const std::string& simplification,
               bool propagate_constants) {
                AGraphExpression expr(simplification,
                                      propagate_constants);
                if (!equation.is_none()) {
                    auto parsing = py::module_::import(
                        "bingo.expressions.agraph.pyagraph.parsing");
                    auto result = parsing.attr(
                        "eq_string_to_command_array_and_constants")(
                        py::str(equation));
                    auto tup = result.cast<py::tuple>();

                    // Command array
                    auto cmd = numpy_to_stack(
                        tup[0].cast<py::array_t<uint8_t>>());
                    expr.set_raw_command_array(std::move(cmd));

                    // Constants
                    expr.set_raw_constants(
                        iterable_to_double_vec(tup[1]));

                    // Integers
                    expr.set_raw_integers(
                        iterable_to_int_vec(tup[2]));
                }
                return expr;
            }),
            py::arg("equation") = py::none(),
            py::kw_only(),
            py::arg("simplification") = "cas",
            py::arg("propagate_constants") = false)

        // ---- Raw layer ----

        .def_property(
            "raw_command_array",
            [](AGraphExpression& self) {
                return stack_to_numpy(self.raw_command_array());
            },
            [](AGraphExpression& self,
               py::array_t<uint8_t, py::array::c_style |
                                     py::array::forcecast> arr) {
                self.set_raw_command_array(numpy_to_stack(arr));
            })

        .def_property_readonly(
            "mutable_raw_command_array",
            [](py::object self_obj) -> py::array_t<uint8_t> {
                auto& self =
                    self_obj.cast<AGraphExpression&>();
                auto& cmd =
                    self.mutable_raw_command_array();
                auto rows =
                    static_cast<py::ssize_t>(cmd.rows());
                auto cols =
                    static_cast<py::ssize_t>(cmd.cols());
                // Return a writable numpy view into internal
                // storage.  self_obj is kept as the base to
                // prevent premature GC.
                return py::array_t<uint8_t>(
                    {rows, cols},
                    {cols * py::ssize_t{1}, py::ssize_t{1}},
                    cmd.data(),
                    self_obj);
            })

        .def_property(
            "raw_constants",
            [](AGraphExpression& self) {
                return vec_to_float_tuple(
                    self.raw_constants());
            },
            [](AGraphExpression& self, py::object val) {
                self.set_raw_constants(
                    iterable_to_double_vec(val));
            })

        .def_property(
            "raw_integers",
            [](AGraphExpression& self) {
                return vec_to_int_tuple(self.raw_integers());
            },
            [](AGraphExpression& self, py::object val) {
                self.set_raw_integers(
                    iterable_to_int_vec(val));
            })

        // ---- Simplified layer ----

        .def_property_readonly(
            "command_array",
            [](AGraphExpression& self) {
                return stack_to_numpy(self.command_array());
            })

        .def_property(
            "constants",
            [](AGraphExpression& self) {
                return vec_to_float_tuple(self.constants());
            },
            [](AGraphExpression& self, py::object val) {
                self.set_constants(
                    iterable_to_double_vec(val));
            })

        .def_property_readonly(
            "integers",
            [](AGraphExpression& self) {
                return vec_to_int_tuple(self.integers());
            })

        .def_property_readonly(
            "constant_mapping",
            [](AGraphExpression& self) {
                return vec_to_int_tuple(
                    self.constant_mapping());
            })

        .def_property_readonly(
            "complexity",
            [](AGraphExpression& self) {
                return self.complexity();
            })

        .def_property_readonly(
            "tree_complexity",
            [](AGraphExpression& self) {
                return self.tree_complexity();
            })

        // ---- Simplification control ----

        .def_property_readonly(
            "simplification",
            [](AGraphExpression& self) {
                return self.simplification();
            })

        .def_property(
            "propagate_constants",
            &AGraphExpression::propagate_constants,
            &AGraphExpression::set_propagate_constants)

        // ---- Format properties (delegate to pyagraph) ----

        .def_property_readonly("console",
            [](AGraphExpression& self) {
                auto& cmd = self.command_array();
                auto formatting = py::module_::import(
                    "bingo.expressions.agraph.pyagraph"
                    ".formatting");
                return formatting.attr(
                    "get_formatted_string")(
                    "console",
                    stack_to_numpy(cmd),
                    vec_to_float_tuple(self.constants()),
                    vec_to_int_tuple(self.integers()));
            })

        .def_property_readonly("sympy",
            [](AGraphExpression& self) {
                auto& cmd = self.command_array();
                auto formatting = py::module_::import(
                    "bingo.expressions.agraph.pyagraph"
                    ".formatting");
                auto sympy_mod =
                    py::module_::import("sympy");
                auto s = formatting.attr(
                    "get_formatted_string")(
                    "sympy",
                    stack_to_numpy(cmd),
                    vec_to_float_tuple(self.constants()),
                    vec_to_int_tuple(self.integers()));
                return sympy_mod.attr("sympify")(s);
            })

        .def_property_readonly("latex",
            [](AGraphExpression& self) {
                auto& cmd = self.command_array();
                auto formatting = py::module_::import(
                    "bingo.expressions.agraph.pyagraph"
                    ".formatting");
                return formatting.attr(
                    "get_formatted_string")(
                    "latex",
                    stack_to_numpy(cmd),
                    vec_to_float_tuple(self.constants()),
                    vec_to_int_tuple(self.integers()));
            })

        // ---- Evaluation ----

        .def("_evaluate", &AGraphExpression::evaluate,
             py::arg("x"))
        .def("_evaluate_with_x_gradient",
             &AGraphExpression::evaluate_with_x_gradient,
             py::arg("x"))
        .def("_evaluate_with_const_gradient",
             &AGraphExpression::evaluate_with_const_gradient,
             py::arg("x"))

        // ---- sklearn interface ----

        .def("predict", &AGraphExpression::predict, py::arg("X"))

        .def("gradient", &AGraphExpression::gradient, py::arg("X"))

        .def("fit",
            [](AGraphExpression& self,
               const RowMatrixXd& X,
               const Eigen::VectorXd& y,
               double tolerance) -> AGraphExpression& {
                self.fit(X, y, tolerance);
                return self;
            },
            py::arg("X"), py::arg("y"),
            py::kw_only(),
            py::arg("tolerance") = 1e-5,
            py::return_value_policy::reference_internal)

        .def("fit_implicit",
            [](AGraphExpression& self,
               const RowMatrixXd& X,
               const RowMatrixXd& dx_dt,
               double tolerance) -> AGraphExpression& {
                self.fit_implicit(X, dx_dt, tolerance);
                return self;
            },
            py::arg("X"), py::arg("dx_dt"),
            py::kw_only(),
            py::arg("tolerance") = 1e-5,
            py::return_value_policy::reference_internal)

        .def("loss", &AGraphExpression::loss,
             py::arg("X"), py::arg("y"),
             py::kw_only(),
             py::arg("kind") = "mse")

        .def("score", &AGraphExpression::score,
             py::arg("X"), py::arg("y"),
             py::kw_only(),
             py::arg("kind") = "r2")

        .def("implicit_loss",
            [](AGraphExpression& self,
               const RowMatrixXd& X,
               const RowMatrixXd& dx_dt,
               py::object required_params) {
                std::optional<int> req;
                if (!required_params.is_none())
                    req = required_params.cast<int>();
                return self.implicit_loss(X, dx_dt, req);
            },
            py::arg("X"), py::arg("dx_dt"),
            py::kw_only(),
            py::arg("required_params") = py::none())

        .def("implicit_score",
            [](AGraphExpression& self,
               const RowMatrixXd& X,
               const RowMatrixXd& dx_dt,
               py::object required_params) {
                std::optional<int> req;
                if (!required_params.is_none())
                    req = required_params.cast<int>();
                return self.implicit_score(X, dx_dt, req);
            },
            py::arg("X"), py::arg("dx_dt"),
            py::kw_only(),
            py::arg("required_params") = py::none())

        .def_property_readonly("is_fitted",
             &AGraphExpression::is_fitted)

        .def("__sklearn_is_fitted__",
             &AGraphExpression::is_fitted)

        // ---- Utility ----

        .def("get_utilized_commands",
             &AGraphExpression::get_utilized_commands)

        .def("promote_simplification",
            [](AGraphExpression& self) -> AGraphExpression& {
                self.promote_simplification();
                return self;
            },
            py::return_value_policy::reference_internal)

        .def("get_operator_counts",
             &AGraphExpression::get_operator_counts,
             py::arg("tree") = true,
             py::arg("terminals") = "exclude")

        .def("distance",
             &AGraphExpression::distance,
             py::arg("other"))

        .def("copy",
            [](AGraphExpression& self) {
                return self.copy();
            })

        // ---- Hash / equality ----

        .def("__hash__",
            [](AGraphExpression& self) {
                return static_cast<py::ssize_t>(self.hash());
            })

        .def("__eq__",
            [](AGraphExpression& self, py::object other) -> py::object {
                if (!py::isinstance<AGraphExpression>(other))
                    return py::reinterpret_borrow<py::object>(
                        Py_NotImplemented);
                auto& o = other.cast<AGraphExpression&>();
                return py::cast(self.equals(o));
            })

        // ---- String representation ----

        .def("__str__",
            [](AGraphExpression& self) {
                auto& cmd = self.command_array();
                auto formatting = py::module_::import(
                    "bingo.expressions.agraph.pyagraph"
                    ".formatting");
                return formatting.attr(
                    "get_formatted_string")(
                    "console",
                    stack_to_numpy(cmd),
                    vec_to_float_tuple(self.constants()),
                    vec_to_int_tuple(self.integers()))
                    .cast<std::string>();
            })

        .def("__repr__",
            [](AGraphExpression& self) {
                auto& cmd = self.command_array();
                auto formatting = py::module_::import(
                    "bingo.expressions.agraph.pyagraph"
                    ".formatting");
                auto s = formatting.attr(
                    "get_formatted_string")(
                    "console",
                    stack_to_numpy(cmd),
                    vec_to_float_tuple(self.constants()),
                    vec_to_int_tuple(self.integers()))
                    .cast<std::string>();
                return "AGraphExpression(" + s + ")";
            })

        // ---- Pickle (getstate / setstate) ----

        .def("__getstate__",
            [](AGraphExpression& self) {
                py::dict state;
                state["_simplification"] =
                    self.simplification();
                state["_propagate_constants"] =
                    self.propagate_constants();
                state["_raw_command_array"] =
                    stack_to_numpy(self.raw_command_array());
                state["_raw_constants"] =
                    vec_to_float_tuple(self.raw_constants());
                state["_raw_integers"] =
                    vec_to_int_tuple(self.raw_integers());
                // Preserve the structure-only fitted lifecycle exactly.
                state["_fit_attempted"] = self.fit_attempted();
                // Fitted constants only need serialization when fitting moved
                // them away from the raw values.
                if (self.fit_attempted() &&
                    self.constants() != self.raw_constants()) {
                    state["_constants"] =
                        vec_to_float_tuple(self.constants());
                }
                return state;
            })

        .def("__setstate__",
            [](AGraphExpression& self, py::dict state) {
                std::string simp =
                    state["_simplification"].cast<std::string>();
                bool propagate =
                    state.contains("_propagate_constants")
                    ? state["_propagate_constants"].cast<bool>()
                    : false;

                new (&self)
                    AGraphExpression(simp, propagate);

                // Setting the raw layer clears the fitted flag; restore it
                // afterwards to preserve the structure-only lifecycle.
                self.set_raw_command_array(
                    numpy_to_stack(
                        state["_raw_command_array"]
                            .cast<py::array_t<uint8_t>>()));

                self.set_raw_constants(
                    iterable_to_double_vec(
                        state["_raw_constants"]));

                self.set_raw_integers(
                    iterable_to_int_vec(
                        state["_raw_integers"]));

                // Backward compat: older pickles encoded fittedness by the
                // mere presence of "_constants".
                bool fit_attempted =
                    state.contains("_fit_attempted")
                        ? state["_fit_attempted"].cast<bool>()
                        : state.contains("_constants");

                // Restore any fitted constants (forces the simplified layer to
                // derive first, then overrides with the stored values).
                if (state.contains("_constants")) {
                    self.set_constants(
                        iterable_to_double_vec(
                            state["_constants"]));
                }

                self.set_fit_attempted(fit_attempted);
            })

        // ---- Deep copy ----

        .def("__deepcopy__",
            [](AGraphExpression& self, py::dict /*memo*/) {
                return self.copy();
            },
            py::arg("memo") = py::dict())

        // ---- Internal state (for test compatibility) ----

        .def_property_readonly("_modified",
            &AGraphExpression::modified)

        .def_property_readonly("_hash",
            [](AGraphExpression& self) -> py::object {
                // Return None if no cached hash
                // (matches Python's self._hash is None check)
                return py::none();
            });
}
