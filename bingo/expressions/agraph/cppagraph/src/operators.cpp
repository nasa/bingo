/**
 * @file operators.cpp
 * @brief Runtime helpers for operator properties and name tables.
 */

#include "cppagraph/operators.h"

namespace cppagraph {

const std::unordered_set<uint8_t>& terminal_ids() {
    static const std::unordered_set<uint8_t> ids{
        static_cast<uint8_t>(Op::VARIABLE),
        static_cast<uint8_t>(Op::CONSTANT),
        static_cast<uint8_t>(Op::INTEGER),
    };
    return ids;
}

const std::unordered_set<uint8_t>& arity_2_ids() {
    static const std::unordered_set<uint8_t> ids{
        static_cast<uint8_t>(Op::ADDITION),
        static_cast<uint8_t>(Op::SUBTRACTION),
        static_cast<uint8_t>(Op::MULTIPLICATION),
        static_cast<uint8_t>(Op::DIVISION),
        static_cast<uint8_t>(Op::POWER),
        static_cast<uint8_t>(Op::SAFE_POWER),
    };
    return ids;
}

const std::unordered_map<uint8_t, std::vector<std::string>>& operator_names() {
    static const std::unordered_map<uint8_t, std::vector<std::string>> names{
        {static_cast<uint8_t>(Op::VARIABLE),       {"load", "x"}},
        {static_cast<uint8_t>(Op::CONSTANT),       {"constant", "c"}},
        {static_cast<uint8_t>(Op::INTEGER),        {"integer"}},
        {static_cast<uint8_t>(Op::ADDITION),       {"add", "addition", "+"}},
        {static_cast<uint8_t>(Op::SUBTRACTION),    {"subtract", "subtraction", "-"}},
        {static_cast<uint8_t>(Op::MULTIPLICATION), {"multiply", "multiplication", "*"}},
        {static_cast<uint8_t>(Op::DIVISION),       {"divide", "division", "/"}},
        {static_cast<uint8_t>(Op::POWER),          {"power", "pow", "^"}},
        {static_cast<uint8_t>(Op::SAFE_POWER),     {"safe power", "safe pow"}},
        {static_cast<uint8_t>(Op::SQUARE),         {"square", "sq"}},
        {static_cast<uint8_t>(Op::CUBE),           {"cube", "cb"}},
        {static_cast<uint8_t>(Op::SQRT),           {"square root", "sqrt"}},
        {static_cast<uint8_t>(Op::ABS),            {"absolute value", "||", "|"}},
        {static_cast<uint8_t>(Op::EXPONENTIAL),    {"exponential", "exp", "e"}},
        {static_cast<uint8_t>(Op::LOGARITHM),      {"logarithm", "log"}},
        {static_cast<uint8_t>(Op::SIN),            {"sine", "sin"}},
        {static_cast<uint8_t>(Op::COS),            {"cosine", "cos"}},
        {static_cast<uint8_t>(Op::TAN),            {"tangent", "tan"}},
        {static_cast<uint8_t>(Op::ARCSIN),         {"arcsin", "asin"}},
        {static_cast<uint8_t>(Op::ARCCOS),         {"arccos", "acos"}},
        {static_cast<uint8_t>(Op::ARCTAN),         {"arctan", "atan"}},
        {static_cast<uint8_t>(Op::SINH),           {"sineh", "sinh"}},
        {static_cast<uint8_t>(Op::COSH),           {"cosineh", "cosh"}},
        {static_cast<uint8_t>(Op::TANH),           {"tangenth", "tanh"}},
    };
    return names;
}

}  // namespace cppagraph
