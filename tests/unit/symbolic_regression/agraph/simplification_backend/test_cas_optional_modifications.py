from bingo.symbolic_regression.agraph.operator_definitions import *
from bingo.symbolic_regression.agraph.simplification_backend.expression import Expression
from bingo.symbolic_regression.agraph.simplification_backend.optional_expression_modification import optional_modifications


def test_inserting_subtraction():
    negative_one = Expression(INTEGER, [-1])
    x = [Expression(VARIABLE, [i, ]) for i in range(4)]
    neg_x = [Expression(MULTIPLICATION, [negative_one, xx]) for xx in x]
    x01_plus_negx23 = Expression(ADDITION, x[:2] + neg_x[2:])
    x01_minus_x23 = Expression(SUBTRACTION,
                               [Expression(ADDITION, x[:2]),
                                Expression(ADDITION, x[2:])])
    assert optional_modifications(x01_plus_negx23) == x01_minus_x23


def test_inserting_subtraction_no_addition():
    negative_one = Expression(INTEGER, [-1])
    x = [Expression(VARIABLE, [i, ]) for i in range(2)]
    neg_x = [Expression(MULTIPLICATION, [negative_one, xx]) for xx in x]
    nx0_plus_nx1 = Expression(ADDITION, neg_x)
    neg_x01 = Expression(MULTIPLICATION,
                         [negative_one,
                          Expression(ADDITION, x)])
    assert optional_modifications(nx0_plus_nx1) == neg_x01


def test_round_trip_with_integer_power():
    x_to_5 = Expression(POWER,
                        [Expression(VARIABLE, [0]),
                         Expression(INTEGER, [5])])
    xxxxx = Expression(MULTIPLICATION, [Expression(VARIABLE, [0])] * 5)
    assert optional_modifications(x_to_5) == xxxxx


def test_power_2_to_multiplication():
    """Test that x^2 converts to multiplication (not square) until bingocpp support"""
    x = Expression(VARIABLE, [0])
    x_to_2 = Expression(POWER, [x, Expression(INTEGER, [2])])
    expected = Expression(MULTIPLICATION, [x, x])
    assert optional_modifications(x_to_2) == expected


def test_power_3_to_multiplication():
    """Test that x^3 converts to multiplication (not cube) until bingocpp support"""
    x = Expression(VARIABLE, [0])
    x_to_3 = Expression(POWER, [x, Expression(INTEGER, [3])])
    expected = Expression(MULTIPLICATION, [x, x, x])
    assert optional_modifications(x_to_3) == expected


def test_power_to_multiplication_for_4():
    """Test that x^4 converts to multiplication"""
    x = Expression(VARIABLE, [0])
    x_to_4 = Expression(POWER, [x, Expression(INTEGER, [4])])
    expected = Expression(MULTIPLICATION, [x] * 4)
    assert optional_modifications(x_to_4) == expected
