/**
 * @file interpreter.cpp
 * @brief Stack ↔ CAS tree translation — port of interpreter.py.
 */

#include "cppagraph/interpreter.h"
#include "cppagraph/automatic_simplification.h"

#include <map>
#include <set>
#include <tuple>
#include <unordered_map>

namespace cppagraph {

static inline uint8_t u8(Op o) { return static_cast<uint8_t>(o); }

// ================================================================== //
//  build_cas_expression                                               //
// ================================================================== //

static CASExprPtr _build_expression_recursive(
    const StackMatrix& stack,
    const std::vector<double>& constants,
    const std::vector<int>& integers,
    int location,
    std::unordered_map<int, CASExprPtr>& memo,
    std::set<int>& visiting)
{
    auto it = memo.find(location);
    if (it != memo.end()) return it->second;

    // Cycle guard.
    if (visiting.count(location)) {
        auto result = interned_variable(0);
        memo[location] = result;
        return result;
    }
    visiting.insert(location);

    uint8_t op     = stack(location, 0);
    int     param1 = stack(location, 1);
    int     param2 = stack(location, 2);

    CASExprPtr result;

    if (IS_TERMINAL[op]) {
        if (op == u8(Op::CONSTANT)) {
            result = std::make_shared<CASExpression>(u8(Op::CONSTANT), param1);
        } else if (op == u8(Op::INTEGER)) {
            int value = (param1 < static_cast<int>(integers.size()))
                        ? integers[param1] : 0;
            result = interned_integer(value);
        } else {
            result = interned_variable(param1);
        }
        memo[location] = result;
        visiting.erase(location);
        return result;
    }

    std::vector<CASExprPtr> operands;
    operands.push_back(
        _build_expression_recursive(stack, constants, integers, param1, memo, visiting));
    if (IS_ARITY_2[op]) {
        operands.push_back(
            _build_expression_recursive(stack, constants, integers, param2, memo, visiting));
    }

    // Normalise SQUARE/CUBE into POWER.
    if (op == u8(Op::SQUARE)) {
        result = std::make_shared<CASExpression>(
            u8(Op::POWER),
            std::vector<CASExprPtr>{operands[0], interned_integer(2)});
    } else if (op == u8(Op::CUBE)) {
        result = std::make_shared<CASExpression>(
            u8(Op::POWER),
            std::vector<CASExprPtr>{operands[0], interned_integer(3)});
    } else {
        result = std::make_shared<CASExpression>(op, std::move(operands));
    }

    memo[location] = result;
    visiting.erase(location);
    return result;
}

CASExprPtr build_cas_expression(
    const StackMatrix& stack,
    const std::vector<double>& constants,
    const std::vector<int>& integers)
{
    std::unordered_map<int, CASExprPtr> memo;
    std::set<int> visiting;
    return _build_expression_recursive(
        stack, constants, integers, static_cast<int>(stack.rows()) - 1, memo, visiting);
}

// ================================================================== //
//  build_simplified_cas_expression                                    //
// ================================================================== //

static CASExprPtr _build_simplified_recursive(
    const StackMatrix& stack,
    const std::vector<double>& constants,
    const std::vector<int>& integers,
    int location,
    std::unordered_map<int, CASExprPtr>& memo,
    const std::unordered_map<uint8_t, SimplifyFn>& simp_funcs,
    std::set<int>& visiting)
{
    auto it = memo.find(location);
    if (it != memo.end()) return it->second;

    if (visiting.count(location)) {
        auto result = interned_variable(0);
        memo[location] = result;
        return result;
    }
    visiting.insert(location);

    uint8_t op     = stack(location, 0);
    int     param1 = stack(location, 1);
    int     param2 = stack(location, 2);

    CASExprPtr result;

    if (IS_TERMINAL[op]) {
        if (op == u8(Op::CONSTANT)) {
            result = std::make_shared<CASExpression>(u8(Op::CONSTANT), param1);
        } else if (op == u8(Op::INTEGER)) {
            int value = (param1 < static_cast<int>(integers.size()))
                        ? integers[param1] : 0;
            result = interned_integer(value);
        } else {
            result = interned_variable(param1);
        }
        memo[location] = result;
        visiting.erase(location);
        return result;
    }

    // Operands are already simplified (bottom-up + memo).
    std::vector<CASExprPtr> operands;
    operands.push_back(
        _build_simplified_recursive(
            stack, constants, integers, param1, memo, simp_funcs, visiting));
    if (IS_ARITY_2[op]) {
        operands.push_back(
            _build_simplified_recursive(
                stack, constants, integers, param2, memo, simp_funcs, visiting));
    }

    // Normalise SQUARE/CUBE into POWER, then simplify.
    uint8_t effective_op = op;
    if (op == u8(Op::SQUARE)) {
        auto node = std::make_shared<CASExpression>(
            u8(Op::POWER),
            std::vector<CASExprPtr>{operands[0], interned_integer(2)});
        effective_op = u8(Op::POWER);
        auto sit = simp_funcs.find(effective_op);
        result = (sit != simp_funcs.end()) ? sit->second(node) : node;
    } else if (op == u8(Op::CUBE)) {
        auto node = std::make_shared<CASExpression>(
            u8(Op::POWER),
            std::vector<CASExprPtr>{operands[0], interned_integer(3)});
        effective_op = u8(Op::POWER);
        auto sit = simp_funcs.find(effective_op);
        result = (sit != simp_funcs.end()) ? sit->second(node) : node;
    } else {
        auto node = std::make_shared<CASExpression>(op, std::move(operands));
        auto sit = simp_funcs.find(op);
        result = (sit != simp_funcs.end()) ? sit->second(node) : node;
    }

    memo[location] = result;
    visiting.erase(location);
    return result;
}

CASExprPtr build_simplified_cas_expression(
    const StackMatrix& stack,
    const std::vector<double>& constants,
    const std::vector<int>& integers)
{
    auto& simp_funcs = simplification_functions();
    std::unordered_map<int, CASExprPtr> memo;
    std::set<int> visiting;
    return _build_simplified_recursive(
        stack, constants, integers, static_cast<int>(stack.rows()) - 1,
        memo, simp_funcs, visiting);
}

// ================================================================== //
//  build_agraph_stack                                                 //
// ================================================================== //

using CommandKey = std::tuple<uint8_t, int, int>;

static int _add_command_to_stack_dict(
    const CommandKey& command,
    std::map<CommandKey, int>& stack_dict)
{
    auto it = stack_dict.find(command);
    if (it != stack_dict.end()) return it->second;
    int loc = static_cast<int>(stack_dict.size());
    stack_dict[command] = loc;
    return loc;
}

static int _add_associative_operators_to_stack(
    uint8_t op,
    const std::vector<int>& operand_locs,
    std::map<CommandKey, int>& stack_dict)
{
    if (operand_locs.size() == 1) return operand_locs[0];
    size_t div = operand_locs.size() / 2;
    std::vector<int> left(operand_locs.begin(), operand_locs.begin() + div);
    std::vector<int> right(operand_locs.begin() + div, operand_locs.end());
    int loc1 = _add_associative_operators_to_stack(op, left, stack_dict);
    int loc2 = _add_associative_operators_to_stack(op, right, stack_dict);
    return _add_command_to_stack_dict({op, loc1, loc2}, stack_dict);
}

static int _build_stack_recursive(
    const CASExprPtr& expression,
    std::map<CommandKey, int>& stack_dict,
    const std::vector<double>& original_constants,
    std::vector<double>& const_list,
    std::vector<int>& int_list,
    std::map<int, int>& const_idx_map,
    std::map<int, int>& int_val_map)
{
    uint8_t op = expression->op();

    if (op == u8(Op::CONSTANT)) {
        int old_idx = expression->terminal_param();
        auto it = const_idx_map.find(old_idx);
        int new_idx;
        if (it != const_idx_map.end()) {
            new_idx = it->second;
        } else {
            new_idx = static_cast<int>(const_list.size());
            double value = (old_idx < static_cast<int>(original_constants.size()))
                           ? original_constants[old_idx] : 1.0;
            const_list.push_back(value);
            const_idx_map[old_idx] = new_idx;
        }
        return _add_command_to_stack_dict({op, new_idx, new_idx}, stack_dict);
    }

    if (op == u8(Op::INTEGER)) {
        int value = expression->terminal_param();
        auto it = int_val_map.find(value);
        int new_idx;
        if (it != int_val_map.end()) {
            new_idx = it->second;
        } else {
            new_idx = static_cast<int>(int_list.size());
            int_list.push_back(value);
            int_val_map[value] = new_idx;
        }
        return _add_command_to_stack_dict({op, new_idx, new_idx}, stack_dict);
    }

    if (op == u8(Op::VARIABLE)) {
        int col = expression->terminal_param();
        return _add_command_to_stack_dict({op, col, col}, stack_dict);
    }

    // Non-terminal.
    auto& operands = expression->operands();
    std::vector<int> operand_locations;
    operand_locations.reserve(operands.size());
    for (auto& child : operands) {
        operand_locations.push_back(
            _build_stack_recursive(
                child, stack_dict, original_constants,
                const_list, int_list, const_idx_map, int_val_map));
    }

    if (operand_locations.size() == 1) {
        return _add_command_to_stack_dict(
            {op, operand_locations[0], operand_locations[0]}, stack_dict);
    }

    if (operand_locations.size() == 2) {
        return _add_command_to_stack_dict(
            {op, operand_locations[0], operand_locations[1]}, stack_dict);
    }

    // Associative operators with >2 operands.
    if (!expression->is_constant_valued()
        && operands[0]->is_constant_valued()) {
        std::vector<int> rest(operand_locations.begin() + 1, operand_locations.end());
        int loc = _add_associative_operators_to_stack(op, rest, stack_dict);
        return _add_command_to_stack_dict({op, operand_locations[0], loc}, stack_dict);
    }

    return _add_associative_operators_to_stack(op, operand_locations, stack_dict);
}

BuildStackResult build_agraph_stack(
    const CASExprPtr& expression,
    const std::vector<double>& original_constants)
{
    std::map<CommandKey, int> stack_dict;
    std::vector<double> const_list;
    std::vector<int> int_list;
    std::map<int, int> const_idx_map;
    std::map<int, int> int_val_map;

    _build_stack_recursive(
        expression, stack_dict, original_constants,
        const_list, int_list, const_idx_map, int_val_map);

    int n = static_cast<int>(stack_dict.size());
    StackMatrix stack(n, 3);
    for (auto& [cmd, loc] : stack_dict) {
        stack(loc, 0) = std::get<0>(cmd);
        stack(loc, 1) = static_cast<uint8_t>(std::get<1>(cmd));
        stack(loc, 2) = static_cast<uint8_t>(std::get<2>(cmd));
    }

    return {std::move(stack), std::move(const_list),
            std::move(int_list), std::move(const_idx_map)};
}

}  // namespace cppagraph
