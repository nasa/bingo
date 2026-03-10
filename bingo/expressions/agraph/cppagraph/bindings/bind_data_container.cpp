/**
 * @file bind_data_container.cpp
 * @brief pybind11 bindings for the DataContainer class.
 *
 * Uses Eigen↔NumPy automatic conversion (pybind11/eigen.h) for
 * zero-copy or low-cost transfers of matrix data.
 */

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "cppagraph/data_container.h"

namespace py = pybind11;
using namespace cppagraph;

void bind_data_container(py::module_& m) {
    py::class_<DataContainer>(m, "DataContainer",
        "Container for training/test data (x, y).\n\n"
        "Parameters\n"
        "----------\n"
        "x : array-like\n"
        "    Feature data, shape (M, D).\n"
        "y : array-like\n"
        "    Target data, shape (M, K).\n\n"
        "Raises\n"
        "------\n"
        "ValueError\n"
        "    If the number of rows in x and y don't match.")

        .def(py::init([](py::array_t<double, py::array::c_style | py::array::forcecast> x_arr,
                         py::array_t<double, py::array::c_style | py::array::forcecast> y_arr) {
                 // Handle 1-D → column vector, matching pyagraph behavior.
                 auto xbuf = x_arr.request();
                 auto ybuf = y_arr.request();

                 RowMatrixXd x, y;

                 if (xbuf.ndim == 1) {
                     x = Eigen::Map<const Eigen::VectorXd>(
                             static_cast<const double*>(xbuf.ptr),
                             xbuf.shape[0]);
                     // Map gives a column vector; we want (N,1) RowMajor
                 } else if (xbuf.ndim == 2) {
                     x = Eigen::Map<const RowMatrixXd>(
                             static_cast<const double*>(xbuf.ptr),
                             xbuf.shape[0], xbuf.shape[1]);
                 } else {
                     throw std::invalid_argument("x must be 1-D or 2-D");
                 }

                 if (ybuf.ndim == 1) {
                     y = Eigen::Map<const Eigen::VectorXd>(
                             static_cast<const double*>(ybuf.ptr),
                             ybuf.shape[0]);
                 } else if (ybuf.ndim == 2) {
                     y = Eigen::Map<const RowMatrixXd>(
                             static_cast<const double*>(ybuf.ptr),
                             ybuf.shape[0], ybuf.shape[1]);
                 } else {
                     throw std::invalid_argument("y must be 1-D or 2-D");
                 }

                 return DataContainer(std::move(x), std::move(y));
             }),
             py::arg("x"), py::arg("y"))

        .def_property_readonly("x",
            [](const DataContainer& dc) -> py::array_t<double> {
                const auto& mat = dc.x();
                // Return a NumPy view (no copy) with shape (M, D).
                return py::array_t<double>(
                    {mat.rows(), mat.cols()},
                    {static_cast<py::ssize_t>(mat.cols() * sizeof(double)),
                     static_cast<py::ssize_t>(sizeof(double))},
                    mat.data(),
                    py::cast(dc)  // prevent gc while array exists
                );
            },
            "Feature data (M x D) as a NumPy array.")

        .def_property_readonly("y",
            [](const DataContainer& dc) -> py::array_t<double> {
                const auto& mat = dc.y();
                return py::array_t<double>(
                    {mat.rows(), mat.cols()},
                    {static_cast<py::ssize_t>(mat.cols() * sizeof(double)),
                     static_cast<py::ssize_t>(sizeof(double))},
                    mat.data(),
                    py::cast(dc)
                );
            },
            "Target data (M x K) as a NumPy array.")

        .def("__len__", &DataContainer::size,
             "Number of data points.")

        .def("__getitem__",
            [](const DataContainer& dc, py::object idx) -> DataContainer {
                // Support integer indexing, slices, and array indexing
                // by converting to a NumPy index on x and y.
                auto x_np = py::array_t<double>(
                    {dc.x().rows(), dc.x().cols()},
                    {static_cast<py::ssize_t>(dc.x().cols() * sizeof(double)),
                     static_cast<py::ssize_t>(sizeof(double))},
                    dc.x().data());
                auto y_np = py::array_t<double>(
                    {dc.y().rows(), dc.y().cols()},
                    {static_cast<py::ssize_t>(dc.y().cols() * sizeof(double)),
                     static_cast<py::ssize_t>(sizeof(double))},
                    dc.y().data());

                py::object x_sliced = x_np.attr("__getitem__")(idx);
                py::object y_sliced = y_np.attr("__getitem__")(idx);

                // Construct a new DataContainer from the sliced arrays.
                auto np = py::module_::import("numpy");
                auto x_2d = np.attr("atleast_2d")(x_sliced);
                auto y_2d = np.attr("atleast_2d")(y_sliced);

                return DataContainer(
                    x_2d.cast<RowMatrixXd>(),
                    y_2d.cast<RowMatrixXd>()
                );
            },
            py::arg("idx"),
            "Row-slice the data container.")
    ;
}
