import re
import numpy as np

from bingo.symbolic_regression.agraph.agraph import AGraph
from bingo.symbolic_regression.agraph.operator_definitions import *

operators = {"+", "-", "*", "/", "^"}
functions = {"sin", "cos", "sinh", "cosh", "exp", "log", "abs", "sqrt"}
precedence = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
operator_map = {"+": ADDITION, "-": SUBTRACTION, "*": MULTIPLICATION,
                "/": DIVISION, "^": POWER, "X": VARIABLE, "C": CONSTANT,
                "sin": SIN, "cos": COS, "sinh": SINH, "cosh": COSH,
                "exp": EXPONENTIAL, "log": LOGARITHM, "abs": ABS,
                "sqrt": SQRT}
var_or_const_pattern = re.compile(r"([XC])_(\d+)")
non_unary_op_pattern = re.compile(r"([*/^()])")


def infix_to_postfix(infix_tokens):  # based on Shunting-yard algorithm
    # can use function version on wikipedia for sin, cos, etc.
    # https://en.wikipedia.org/wiki/Shunting-yard_algorithm
    stack = []  # index -1 = top
    output = []
    for token in infix_tokens:
        if token in operators:
            while len(stack) > 0 and stack[-1] in operators and \
                (precedence[stack[-1]] > precedence[token] or
                 precedence[stack[-1]] == precedence[token] and token != "^"):
                # TODO make test case to make sure left-associativity isn't used with power (e.g. 1.0 + X_0^4.0))
                # also 3 + 4 × 2 ÷ ( 1 − 5 ) ^ 2 ^ 3
                output.append(stack.pop())
            stack.append(token)
        elif token == "(":
            stack.append(token)
        elif token in functions:
            stack.append(token)
        elif token == ")":
            while stack[-1] != "(":
                if len(stack) == 0:
                    raise RuntimeError("Mismatched parenthesis")
                output.append(stack.pop())
            stack.pop()  # get rid of "("
            if len(stack) > 0 and stack[-1] in functions:
                output.append(stack.pop())
        else:
            output.append(token)

    while len(stack) > 0:
        token = stack.pop()
        if token == "(":
            raise RuntimeError("Mismatched parenthesis")
        output.append(token)

    return output


def postfix_to_command_array_and_constants(postfix_tokens):
    stack = []  # -1 = top (the data structure, not a command_array)
    command_array = []
    constants = []
    i = 0
    for token in postfix_tokens:
        if token in operators:
            operands = stack.pop(), stack.pop()
            command_array.append([operator_map[token], operands[1], operands[0]])
        elif token in functions:
            operand = stack.pop()
            command_array.append([operator_map[token], operand, operand])
        else:
            var_or_const = var_or_const_pattern.fullmatch(token)
            if var_or_const:
                groups = var_or_const.groups()
                command_array.append([operator_map[groups[0]], int(groups[1]), int(groups[1])])
            else:
                try:
                    constant = float(token)
                    c_i = len(constants)
                    command_array.append([CONSTANT, c_i, c_i])
                    constants.append(constant)
                except ValueError:
                    raise RuntimeError("Unknown token", token)
        stack.append(i)
        i += 1

    if len(stack) > 1:
        raise RuntimeError("Error evaluating postfix expression")
    return np.array(command_array, dtype=int), constants


def infix_tokens_to_agraph(tokens, use_simplification=False):
    command_array, constants = postfix_to_command_array_and_constants(infix_to_postfix(tokens))
    graph = AGraph(use_simplification)
    graph.command_array = command_array
    graph.set_local_optimization_params(constants)
    return graph


def sympy_string_to_agraph(sympy_string, use_simplification=False):  # TODO change this to a constructor
    sympy_string = sympy_string.replace("**", "^")
    tokens = non_unary_op_pattern.sub(r" \1 ", sympy_string).split(" ")
    tokens = [x for x in tokens if x != ""]  # for if there was a trailing space in sympy_string after sub
    return infix_tokens_to_agraph(tokens, use_simplification)


if __name__ == '__main__':
    # test_graph = AGraph()
    # test_graph.command_array = np.array([[INTEGER, 1, 1]], dtype=int)
    # infix = test_graph.get_formatted_string("infix").split(" ")
    # print(test_graph)

    infix = "4.33 * sin ( X_0 ) + X_1 + log ( 2.554 )".split(" ")
    print(infix)

    postfix = infix_to_postfix(infix)
    print(postfix)

    command_array, constants = postfix_to_command_array_and_constants(postfix)
    print(command_array)
    print(constants)

    output_graph = infix_tokens_to_agraph(infix)
    print(output_graph)
