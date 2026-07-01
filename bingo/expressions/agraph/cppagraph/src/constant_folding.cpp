/**
 * @file constant_folding.cpp
 * @brief Constant folding — port of constant_folding.py.
 */

#include "cppagraph/constant_folding.h"

#include <algorithm>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

namespace cppagraph {

static inline uint8_t u8(Op o) { return static_cast<uint8_t>(o); }

static bool _same_expression(
    const CASExprPtr& lhs,
    const CASExprPtr& rhs)
{
    if (lhs == rhs) return true;
    if (!lhs || !rhs) return !lhs && !rhs;
    return *lhs == *rhs;
}

static bool _same_expression_sequence(
    const std::vector<CASExprPtr>& lhs,
    const std::vector<CASExprPtr>& rhs)
{
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (!_same_expression(lhs[i], rhs[i])) return false;
    }
    return true;
}

struct CASExprPtrStructHash {
    std::size_t operator()(const CASExprPtr& expression) const {
        return expression ? expression->hash() : 0;
    }
};

struct CASExprPtrStructEq {
    bool operator()(const CASExprPtr& lhs, const CASExprPtr& rhs) const {
        return _same_expression(lhs, rhs);
    }
};

using StructuralExprSet = std::unordered_set<
    CASExprPtr,
    CASExprPtrStructHash,
    CASExprPtrStructEq>;

static bool _same_expression_set(
    const StructuralExprSet& lhs,
    const StructuralExprSet& rhs)
{
    if (lhs.size() != rhs.size()) return false;

    for (auto& expression : lhs) {
        if (rhs.find(expression) == rhs.end()) return false;
    }
    return true;
}

static const std::unordered_set<uint8_t> TERMINAL_OPS = {
    u8(Op::CONSTANT), u8(Op::INTEGER), u8(Op::VARIABLE)
};
static const std::unordered_set<uint8_t> ASSOC_OPS = {
    u8(Op::MULTIPLICATION), u8(Op::ADDITION)
};

// Forward declarations.
static CASExprPtr _group_constants(const CASExprPtr& expression);

// ------------------------------------------------------------------ //
// Utility: depends_on as a set of ints (constant indices) + strings   //
// ------------------------------------------------------------------ //

using DepSet = CASExpression::DepSet;
static const DepSet I_SET = {std::string("i")};

// ------------------------------------------------------------------ //
// Fused discovery (single DFS)                                        //
// ------------------------------------------------------------------ //

struct IPData {
    DepSet deps;
    std::vector<CASExprPtr> operands;
};

using ExprIdentity = const CASExpression*;
using IPMap = std::unordered_map<ExprIdentity, IPData>;
using ConstMap = std::unordered_map<int, CASExprPtr>;
using ConstIndexSet = std::unordered_set<int>;

static void _fused_dfs(
    const CASExprPtr& expression,
    ConstMap& constants,
    std::vector<int>& constant_order,
    IPMap& ip_map)
{
    uint8_t op = expression->op();
    if (op == u8(Op::CONSTANT)) {
        auto [it, inserted] = constants.emplace(
            expression->terminal_param(), expression);
        if (!inserted) {
            it->second = expression;
        } else {
            constant_order.push_back(expression->terminal_param());
        }
        return;
    }
    if (TERMINAL_OPS.count(op)) return;

    auto& operands = expression->operands();
    for (auto& child : operands)
        _fused_dfs(child, constants, constant_order, ip_map);

    ip_map[expression.get()] = {expression->depends_on(), operands};
}

// ------------------------------------------------------------------ //
// Subset generation                                                   //
// ------------------------------------------------------------------ //

template <typename Fn>
static void _generate_subsets(
    const std::vector<int>& items,
    size_t subset_size,
    size_t start,
    std::vector<int>& current,
    Fn&& fn,
    bool& stop)
{
    if (stop) return;

    if (current.size() == subset_size) {
        stop = fn(current);
        return;
    }

    size_t remaining = subset_size - current.size();
    if (items.size() - start < remaining) return;

    for (size_t i = start; i < items.size(); ++i) {
        current.push_back(items[i]);
        _generate_subsets(items, subset_size, i + 1, current, fn, stop);
        current.pop_back();
        if (stop) return;
    }
}

template <typename Fn>
static bool for_each_subset(const std::vector<int>& items, Fn&& fn) {
    for (size_t subset_size = 1; subset_size <= items.size(); ++subset_size) {
        std::vector<int> current;
        current.reserve(subset_size);
        bool stop = false;
        _generate_subsets(items, subset_size, 0, current, fn, stop);
        if (stop) return true;
    }
    return false;
}

// ------------------------------------------------------------------ //
// Filter insertion points                                             //
// ------------------------------------------------------------------ //

struct InsertionEntry {
    CASExprPtr parent;       // nullptr means "root replacement"
    std::vector<CASExprPtr> children;
};

struct InsertionBucket {
    CASExprPtr node;
    std::vector<InsertionEntry> entries;
};

using InsertionMap = std::vector<InsertionBucket>;

using InsertionIndexMap = std::unordered_map<
    CASExprPtr,
    std::size_t,
    CASExprPtrStructHash,
    CASExprPtrStructEq>;

static void _add_insertion_entry(
    InsertionMap& insertion_points,
    InsertionIndexMap& insertion_index,
    const CASExprPtr& node,
    InsertionEntry entry)
{
    auto index_it = insertion_index.find(node);
    std::vector<InsertionEntry>* entries = nullptr;
    if (index_it == insertion_index.end()) {
        insertion_points.push_back({node, {}});
        entries = &insertion_points.back().entries;
        insertion_index.emplace(node, insertion_points.size() - 1);
    } else {
        entries = &insertion_points[index_it->second].entries;
    }

    for (auto& existing : *entries) {
        if (_same_expression(existing.parent, entry.parent)
            && _same_expression_sequence(existing.children, entry.children)) {
            return;
        }
    }
    entries->push_back(std::move(entry));
}

static void _filter_ip_recurse(
    const CASExprPtr& expression,
    const ConstIndexSet& const_set,
    const DepSet& const_and_i,
    InsertionMap& insertion_points,
    InsertionIndexMap& insertion_index,
    const IPMap& ip_map,
    const CASExprPtr& parent)
{
    auto it = ip_map.find(expression.get());
    if (it == ip_map.end()) return;  // terminal

    auto& operands = it->second.operands;
    for (auto& operand : operands)
        _filter_ip_recurse(
            operand, const_set, const_and_i, insertion_points,
            insertion_index, ip_map, expression);

    // Check if this node is an insertion point.
    bool any_solely = false;
    bool any_others = false;
    for (auto& operand : operands) {
        auto& od = operand->depends_on();
        // has_consts: od intersects const_set
        bool has_consts = false;
        for (auto& dep : od) {
            if (auto* pi = std::get_if<int>(&dep)) {
                if (const_set.count(*pi)) { has_consts = true; break; }
            }
        }
        // has_other: od is NOT a subset of const_and_i
        bool has_other = false;
        for (auto& dep : od) {
            if (const_and_i.find(dep) == const_and_i.end()) {
                has_other = true; break;
            }
        }
        if (has_consts && !has_other) any_solely = true;
        if (has_other) any_others = true;
        if (any_solely && any_others) break;
    }
    if (!(any_solely && any_others)) return;

    if (expression->is_constant_valued()) {
        _add_insertion_entry(
            insertion_points, insertion_index, expression, {parent, {expression}});
    } else {
        std::vector<CASExprPtr> constant_operands;
        StructuralExprSet seen_operands;
        for (auto& operand : operands) {
            auto& od = operand->depends_on();
            bool subset = true;
            for (auto& dep : od) {
                if (const_and_i.find(dep) == const_and_i.end()) {
                    subset = false; break;
                }
            }
            if (subset && seen_operands.insert(operand).second)
                constant_operands.push_back(operand);
        }
        _add_insertion_entry(
            insertion_points, insertion_index,
            expression, {expression, std::move(constant_operands)});
    }
}

static InsertionMap _filter_insertion_points(
    const CASExprPtr& expression,
    const ConstIndexSet& const_set,
    const IPMap& ip_map)
{
    auto& deps = expression->depends_on();
    DepSet const_and_i;
    for (int c : const_set) const_and_i.insert(c);
    const_and_i.insert(std::string("i"));

    // Check if root depends solely on const_set.
    bool deps_has_const = false;
    for (auto& dep : deps) {
        if (auto* pi = std::get_if<int>(&dep)) {
            if (const_set.count(*pi)) { deps_has_const = true; break; }
        }
    }
    bool deps_subset = true;
    for (auto& dep : deps) {
        if (const_and_i.find(dep) == const_and_i.end()) {
            deps_subset = false; break;
        }
    }

    if (deps_has_const && deps_subset) {
        InsertionMap result;
        result.push_back({expression, {{nullptr, {expression}}}});
        return result;
    }

    InsertionMap insertion_points;
    InsertionIndexMap insertion_index;
    _filter_ip_recurse(
        expression, const_set, const_and_i, insertion_points,
        insertion_index, ip_map, nullptr);
    return insertion_points;
}

// ------------------------------------------------------------------ //
// Generate replacement instructions                                   //
// ------------------------------------------------------------------ //

using ReplacementInnerMap = std::unordered_map<
    CASExprPtr,
    CASExprPtr,
    CASExprPtrStructHash,
    CASExprPtrStructEq>;
using ReplacementMap = std::unordered_map<
    CASExprPtr,
    ReplacementInnerMap,
    CASExprPtrStructHash,
    CASExprPtrStructEq>;
// nullptr value in inner map = "remove this child"

static ReplacementMap _generate_replacement_instructions(
    const std::vector<int>& const_subset,
    const ConstMap& constants,
    const InsertionMap& insertion_points)
{
    if (insertion_points.size() > const_subset.size()) return {};

    ReplacementMap replacements;
    StructuralExprSet constants_to_insert;
    StructuralExprSet expressions_to_replace;

    auto cs_it = const_subset.begin();
    for (auto& bucket : insertion_points) {
        if (cs_it == const_subset.end()) break;
        int const_num = *cs_it++;
        auto cit = constants.find(const_num);
        if (cit == constants.end()) continue;
        auto const_to_insert = cit->second;

        for (auto& entry : bucket.entries) {
            int i = 0;
            for (auto& child : entry.children) {
                expressions_to_replace.insert(child);
                if (i == 0) {
                    replacements[entry.parent][child] = const_to_insert;
                    constants_to_insert.insert(const_to_insert);
                } else {
                    replacements[entry.parent][child] = nullptr;
                    constants_to_insert.insert(nullptr);
                }
                ++i;
            }
        }
    }

    if (_same_expression_set(constants_to_insert, expressions_to_replace)) return {};
    return replacements;
}

// ------------------------------------------------------------------ //
// Perform constant folding                                            //
// ------------------------------------------------------------------ //

static CASExprPtr _perform_constant_folding(
    const CASExprPtr& expression,
    const ReplacementMap& replacements);

static std::vector<CASExprPtr> _get_new_operands_with_replacements(
    const CASExprPtr& expression,
    const ReplacementMap& replacements)
{
    std::vector<CASExprPtr> new_operands;
    auto rit = replacements.find(expression);
    if (rit == replacements.end()) return expression->operands();

    auto& reps = rit->second;
    for (auto& operand : expression->operands()) {
        auto orit = reps.find(operand);
        if (orit != reps.end()) {
            if (orit->second) new_operands.push_back(orit->second);
            // else: nullptr = remove this operand
        } else {
            new_operands.push_back(_perform_constant_folding(operand, replacements));
        }
    }
    return new_operands;
}

static CASExprPtr _recursive_expression_replacement(
    const CASExprPtr& expression,
    const ReplacementMap& replacements)
{
    if (replacements.find(expression) == replacements.end()) {
        if (TERMINAL_OPS.count(expression->op())) return expression;
        return expression->map([&](const CASExprPtr& x) {
            return _perform_constant_folding(x, replacements);
        });
    }
    auto new_operands = _get_new_operands_with_replacements(expression, replacements);
    return std::make_shared<CASExpression>(expression->op(), std::move(new_operands));
}

static CASExprPtr _perform_constant_folding(
    const CASExprPtr& expression,
    const ReplacementMap& replacements)
{
    // Check for "root replacement" (parent = nullptr key).
    auto null_it = replacements.find(nullptr);
    if (null_it != replacements.end()) {
        auto expr_it = null_it->second.find(expression);
        if (expr_it != null_it->second.end()) return expr_it->second;
    }
    return _recursive_expression_replacement(expression, replacements);
}

// ------------------------------------------------------------------ //
// Group constants                                                     //
// ------------------------------------------------------------------ //

static CASExprPtr _group_constants(const CASExprPtr& expression) {
    if (TERMINAL_OPS.count(expression->op())) return expression;

    auto& orig_operands = expression->operands();
    std::vector<CASExprPtr> new_operands;
    new_operands.reserve(orig_operands.size());
    bool changed = false;
    for (auto& op : orig_operands) {
        auto grouped = _group_constants(op);
        new_operands.push_back(grouped);
        if (grouped.get() != op.get()) changed = true;
    }

    if (ASSOC_OPS.count(expression->op())) {
        std::vector<CASExprPtr> const_ops, non_const_ops;
        for (auto& op : new_operands) {
            if (op->is_constant_valued())
                const_ops.push_back(op);
            else
                non_const_ops.push_back(op);
        }
        if (const_ops.size() > 1 && !non_const_ops.empty()) {
            auto const_expr = std::make_shared<CASExpression>(
                expression->op(), std::move(const_ops));
            std::vector<CASExprPtr> combined = {const_expr};
            combined.insert(combined.end(), non_const_ops.begin(), non_const_ops.end());
            return std::make_shared<CASExpression>(expression->op(), std::move(combined));
        }
    }

    if (!changed) return expression;
    return std::make_shared<CASExpression>(expression->op(), std::move(new_operands));
}

// ------------------------------------------------------------------ //
// fold_constants (public)                                             //
// ------------------------------------------------------------------ //

CASExprPtr fold_constants(const CASExprPtr& expression) {
    auto expr = _group_constants(expression);

    bool check_for_folding = true;
    while (check_for_folding) {
        check_for_folding = false;

        ConstMap cas_constants;
        std::vector<int> constant_order;
        IPMap ip_map;
        _fused_dfs(expr, cas_constants, constant_order, ip_map);

        check_for_folding = for_each_subset(constant_order, [&](const std::vector<int>& const_subset) {
            ConstIndexSet const_set(const_subset.begin(), const_subset.end());
            auto insertion_points = _filter_insertion_points(
                expr, const_set, ip_map);
            auto replacements = _generate_replacement_instructions(
                const_subset, cas_constants, insertion_points);
            if (!replacements.empty()) {
                expr = _perform_constant_folding(expr, replacements);
                return true;
            }
            return false;
        });
    }

    return expr;
}

}  // namespace cppagraph
