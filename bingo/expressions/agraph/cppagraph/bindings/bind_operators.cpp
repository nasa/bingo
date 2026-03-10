/**
 * @file bind_operators.cpp
 * @brief pybind11 bindings for operator IDs and property tables.
 *
 * Exposes every symbol that pyagraph.operators exports so that the
 * Python-side __init__.py can do a simple ``from ._cppagraph import *``
 * and present an identical public API.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "cppagraph/operators.h"

namespace py = pybind11;
using namespace cppagraph;

void bind_operators(py::module_& m) {
    // ---- Operator ID integer constants (match pyagraph exactly) ----
    m.attr("VARIABLE")       = static_cast<uint8_t>(Op::VARIABLE);
    m.attr("CONSTANT")       = static_cast<uint8_t>(Op::CONSTANT);
    m.attr("INTEGER")        = static_cast<uint8_t>(Op::INTEGER);
    m.attr("ADDITION")       = static_cast<uint8_t>(Op::ADDITION);
    m.attr("SUBTRACTION")    = static_cast<uint8_t>(Op::SUBTRACTION);
    m.attr("MULTIPLICATION") = static_cast<uint8_t>(Op::MULTIPLICATION);
    m.attr("DIVISION")       = static_cast<uint8_t>(Op::DIVISION);
    m.attr("POWER")          = static_cast<uint8_t>(Op::POWER);
    m.attr("SAFE_POWER")     = static_cast<uint8_t>(Op::SAFE_POWER);
    m.attr("SQUARE")         = static_cast<uint8_t>(Op::SQUARE);
    m.attr("CUBE")           = static_cast<uint8_t>(Op::CUBE);
    m.attr("SQRT")           = static_cast<uint8_t>(Op::SQRT);
    m.attr("ABS")            = static_cast<uint8_t>(Op::ABS);
    m.attr("EXPONENTIAL")    = static_cast<uint8_t>(Op::EXPONENTIAL);
    m.attr("LOGARITHM")      = static_cast<uint8_t>(Op::LOGARITHM);
    m.attr("SIN")            = static_cast<uint8_t>(Op::SIN);
    m.attr("COS")            = static_cast<uint8_t>(Op::COS);
    m.attr("TAN")            = static_cast<uint8_t>(Op::TAN);
    m.attr("ARCSIN")         = static_cast<uint8_t>(Op::ARCSIN);
    m.attr("ARCCOS")         = static_cast<uint8_t>(Op::ARCCOS);
    m.attr("ARCTAN")         = static_cast<uint8_t>(Op::ARCTAN);
    m.attr("SINH")           = static_cast<uint8_t>(Op::SINH);
    m.attr("COSH")           = static_cast<uint8_t>(Op::COSH);
    m.attr("TANH")           = static_cast<uint8_t>(Op::TANH);

    // ---- frozenset equivalents (Python frozenset) ----
    {
        const auto& tids = terminal_ids();
        py::set s;
        for (auto id : tids)
            s.add(py::int_(id));
        m.attr("TERMINAL_IDS") = py::frozenset(s);
    }
    {
        const auto& aids = arity_2_ids();
        py::set s;
        for (auto id : aids)
            s.add(py::int_(id));
        m.attr("ARITY_2_IDS") = py::frozenset(s);
    }

    // ---- NumPy boolean lookup arrays ----
    {
        auto arr = py::array_t<bool>(NUM_OPS);
        auto buf = arr.mutable_unchecked<1>();
        for (std::size_t i = 0; i < NUM_OPS; ++i)
            buf(i) = IS_TERMINAL[i];
        arr.attr("flags").attr("writeable") = false;
        m.attr("IS_TERMINAL_ARRAY") = arr;
    }
    {
        auto arr = py::array_t<bool>(NUM_OPS);
        auto buf = arr.mutable_unchecked<1>();
        for (std::size_t i = 0; i < NUM_OPS; ++i)
            buf(i) = IS_ARITY_2[i];
        arr.attr("flags").attr("writeable") = false;
        m.attr("IS_ARITY_2_ARRAY") = arr;
    }

    // ---- Operator name mapping (dict[int, list[str]]) ----
    {
        py::dict d;
        for (const auto& [id, names] : operator_names()) {
            py::list pynames;
            for (const auto& n : names)
                pynames.append(py::str(n));
            d[py::int_(id)] = pynames;
        }
        m.attr("OPERATOR_NAMES") = d;
    }
}
