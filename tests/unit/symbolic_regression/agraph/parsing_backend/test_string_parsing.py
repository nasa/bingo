import pytest

from bingo.symbolic_regression.agraph.parsing_backend.string_parsing import \
    infix_to_postfix, postfix_to_command_array_and_constants, \
    sympy_string_to_infix_tokens
from bingo.symbolic_regression.agraph.string_generation import get_formatted_string


def test_infix_to_postfix_basic_order_of_operations():
    infix_tokens = list("a+2*b^2-c/3")
    expected_postfix = list("a2b2^*+c3/-")
    assert infix_to_postfix(infix_tokens) == expected_postfix


def test_infix_to_postfix_parenthesis_order_of_operations():
    infix_tokens = list("(a+b)*2+(c-d)/3+e^(f+3)")
    expected_postfix = list("ab+2*cd-3/+ef3+^+")
    assert infix_to_postfix(infix_tokens) == expected_postfix


def test_infix_to_postfix_power_is_right_associative():
    normal_tokens = list("a^b^c")
    explicit_right_associative_tokens = list("a^(b^c)")
    expected_right_associative_postfix = list("abc^^")
    assert infix_to_postfix(normal_tokens) == expected_right_associative_postfix
    assert infix_to_postfix(explicit_right_associative_tokens) == expected_right_associative_postfix

    explicit_left_associative_tokens = list("(a^b)^c")
    expected_left_associative_postfix = list("ab^c^")
    assert infix_to_postfix(explicit_left_associative_tokens) == expected_left_associative_postfix


def test_infix_to_postfix_subtraction_is_left_associative():
    normal_tokens = list("a-b-c")
    explicit_left_associative_tokens = list("(a-b)-c")
    expected_left_associative_postfix = list("ab-c-")
    assert infix_to_postfix(normal_tokens) == expected_left_associative_postfix
    assert infix_to_postfix(explicit_left_associative_tokens) == expected_left_associative_postfix

    explicit_right_associative_tokens = list("a-(b-c)")
    expected_right_associative_postfix = list("abc--")
    assert infix_to_postfix(explicit_right_associative_tokens) == expected_right_associative_postfix


def test_infix_to_postfix_division_is_left_associative():
    normal_tokens = list("a/b/c")
    explicit_left_associative_tokens = list("(a/b)/c")
    expected_left_associative_postfix = list("ab/c/")
    assert infix_to_postfix(normal_tokens) == expected_left_associative_postfix
    assert infix_to_postfix(explicit_left_associative_tokens) == expected_left_associative_postfix

    explicit_right_associative_tokens = list("a/(b/c)")
    expected_right_associative_postfix = list("abc//")
    assert infix_to_postfix(explicit_right_associative_tokens) == expected_right_associative_postfix


def test_infix_to_postfix_addition_is_left_associative():
    # technically addition is fully associative, but assuming it is left
    # associative does not affect expression evaluation
    # (i.e. since (a + b) + c = a + (b + c), we can assume a + b + c = (a + b) + c)
    normal_tokens = list("a+b+c")
    explicit_left_associative_tokens = list("(a+b)+c")
    expected_left_associative_postfix = list("ab+c+")
    assert infix_to_postfix(normal_tokens) == expected_left_associative_postfix
    assert infix_to_postfix(explicit_left_associative_tokens) == expected_left_associative_postfix

    explicit_right_associative_tokens = list("a+(b+c)")
    expected_right_associative_postfix = list("abc++")
    assert infix_to_postfix(explicit_right_associative_tokens) == expected_right_associative_postfix


def test_infix_to_postfix_multiplication_is_left_associative():
    # technically multiplication is fully associative, but assuming it is left
    # associative does not affect expression evaluation
    # (i.e. since (a * b) * c = a * (b * c), we can assume a * b * c = (a * b) * c)
    normal_tokens = list("a*b*c")
    explicit_left_associative_tokens = list("(a*b)*c")
    expected_left_associative_postfix = list("ab*c*")
    assert infix_to_postfix(normal_tokens) == expected_left_associative_postfix
    assert infix_to_postfix(explicit_left_associative_tokens) == expected_left_associative_postfix

    explicit_right_associative_tokens = list("a*(b*c)")
    expected_right_associative_postfix = list("abc**")
    assert infix_to_postfix(explicit_right_associative_tokens) == expected_right_associative_postfix


@pytest.mark.parametrize("expression", ["(a(b)", "a(b))", "a*b)"])
def test_infix_to_postfix_mismatched_parens(expression):
    mismatched_paren_tokens = list(expression)
    with pytest.raises(RuntimeError) as exception_info:
        infix_to_postfix(mismatched_paren_tokens)
    assert str(exception_info.value) == "Mismatched parenthesis"


@pytest.mark.parametrize("expression, expected",
                         [("sin ( a * b ) + cos ( c / d )",
                          "a b * sin c d / cos +"),
                          ("sinh ( a ) * cosh ( b )", "a sinh b cosh *"),
                          ("exp ( a ) ^ log ( b )", "a exp b log ^"),
                          ("abs ( a ) / sqrt ( b )", "a abs b sqrt /"),
                          ("( sin ( a * b ) + cos ( c / d ) ) ^ 2",
                           "a b * sin c d / cos + 2 ^")])
def test_infix_to_postfix_with_functions(expression, expected):
    infix_tokens = expression.split(" ")
    expected_postfix = expected.split(" ")
    assert infix_to_postfix(infix_tokens) == expected_postfix


def test_postfix_to_command_array_and_constants_basic():
    postfix_tokens = "X_0 1.0 + sin X_0 2.0 / cos * 2.0 -".split(" ")
    # sin(X_0 + 1.0) * cos(X_0 / 2.0) - 2.0
    expected_console_string = "(sin(X_0 + 1.0))(cos((X_0)/(2.0) )) - (2.0)"
    command_array, constants = postfix_to_command_array_and_constants(postfix_tokens)
    assert get_formatted_string("console", command_array, constants) == expected_console_string


def test_postfix_to_command_array_and_constants_complex():
    postfix_tokens = "X_0 X_1 + 2.0 * X_2 X_3 - 3 / + X_4 X_5 3 + ^ +".split(" ")
    # (X_0 + X_1) * 2.0 + (X_2 - X_3) / 3 + X_4^(X_5 + 3)
    expected_console_string = "(X_0 + X_1)(2.0) + (X_2 - (X_3))/(3)  + (X_4)^(X_5 + 3)"
    command_array, constants = postfix_to_command_array_and_constants(postfix_tokens)
    assert get_formatted_string("console", command_array, constants) == expected_console_string


def test_postfix_to_command_array_and_constants_unordered_constants():
    postfix_tokens = "C_1 C_0 +".split(" ")  # C_1 + C_0 +
    constants = [1.0, 2.0]
    expected_console_string = "2.0 + 1.0"
    command_array, _ = postfix_to_command_array_and_constants(postfix_tokens)
    assert get_formatted_string("console", command_array, constants) == expected_console_string


def test_postfix_to_command_array_and_constants_unordered_variables():
    postfix_tokens = "X_1 X_0 +".split(" ")
    expected_console_string = "X_1 + X_0"
    command_array, constants = postfix_to_command_array_and_constants(postfix_tokens)
    assert get_formatted_string("console", command_array, constants) == expected_console_string


def test_postfix_to_command_array_and_constants_duplicated_vars_consts_and_ints():
    postfix_tokens = "X_0 1.0 + 2 + X_0 1.0 + 2 + +".split(" ")
    expected_console_string = "X_0 + 1.0 + 2 + X_0 + 1.0 + 2"
    command_array, constants = postfix_to_command_array_and_constants(postfix_tokens)
    assert len(command_array) == 8  # don't duplicate variables, constants, or integers
    assert len(constants) == 1  # don't duplicate constants
    assert get_formatted_string("console", command_array, constants) == expected_console_string


@pytest.mark.parametrize("function_string", ["sin", "cos", "sinh", "cosh", "exp", "log", "abs", "sqrt"])
def test_postfix_to_command_array_functions(function_string):
    postfix_tokens = f"1.0 2.0 + {function_string}".split(" ")  # function(1.0 + 2.0)
    if function_string == "abs":
        expected_console_string = "|1.0 + 2.0|"
    else:
        expected_console_string = f"{function_string}(1.0 + 2.0)"
    command_array, constants = postfix_to_command_array_and_constants(postfix_tokens)
    assert get_formatted_string("console", command_array, constants) == expected_console_string


def test_postfix_to_command_array_and_constants_unknown_token():
    postfix_tokens = "X_0 b +".split(" ")
    with pytest.raises(RuntimeError) as exception_info:
        postfix_to_command_array_and_constants(postfix_tokens)
        assert exception_info.value == "Unknown token b"


@pytest.mark.parametrize("postfix_str, expected_exception, exception_message",
                         [("X_0 +", IndexError, "pop from empty list"),
                          ("X_0 X_1 + X_3", RuntimeError, "Error evaluating postfix expression")])
def test_postfix_to_command_array_and_constants_invalid_postfix(postfix_str, expected_exception, exception_message):
    postfix_tokens = postfix_str.split(" ")
    with pytest.raises(expected_exception) as exception_info:
        postfix_to_command_array_and_constants(postfix_tokens)
        assert exception_info.value == exception_message


def test_sympy_string_to_infix_tokens_basic():
    sympy_string = "-1.0 + X_0 + 2.0"
    assert sympy_string_to_infix_tokens(sympy_string) == sympy_string.split(" ")


def test_sympy_string_to_infix_tokens_complex():
    sympy_string = "X_4**(X_5 + 3) + 2.0*log(X_0 + X_1) + cosh(X_2 - X_3)/3"
    expected_infix_tokens = "X_4 ^ ( X_5 + 3 ) + 2.0 * log ( X_0 + X_1 ) + cosh ( X_2 - X_3 ) / 3".split(" ")
    assert sympy_string_to_infix_tokens(sympy_string) == expected_infix_tokens