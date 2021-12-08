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
integer_pattern = re.compile(r"\d+")


def convert_to_postfix(infix_tokens):  # based on Shunting-yard algorithm
    # can use function version on wikipedia for sin, cos, etc.
    # https://en.wikipedia.org/wiki/Shunting-yard_algorithm
    stack = []  # index -1 = top
    output = []
    for token in infix_tokens:
        if token in operators:
            while len(stack) > 0 and stack[-1] in operators and precedence[stack[-1]] >= precedence[token]:
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
            if stack[-1] in functions:
                output.append(stack.pop())
        else:
            output.append(token)

    for token in stack:
        if token == "(":
            raise RuntimeError("Mismatched parenthesis")
        output.append(token)

    return output


def postfix_to_command_array(postfix_tokens):
    stack = []  # -1 = top (the data structure, not a command_array)
    command_array = []
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
            integer = integer_pattern.fullmatch(token)
            if var_or_const:
                groups = var_or_const.groups()
                command_array.append([operator_map[groups[0]], int(groups[1]), int(groups[1])])
            elif integer:
                operand = int(integer.group(0))
                command_array.append([INTEGER, operand, operand])
            else:
                raise RuntimeError("Unknown token", token)
        stack.append(i)
        i += 1

    if len(stack) > 1:
        raise RuntimeError("Error evaluating postfix expression")
    return command_array


if __name__ == '__main__':
    # test_graph = AGraph()
    # test_graph.command_array = np.array([[INTEGER, 1, 1]], dtype=int)
    # infix = test_graph.get_formatted_string("infix").split(" ")
    # print(test_graph)

    infix = "sin ( X_0 ) + X_1 + log ( C_0 )".split(" ")
    print(infix)

    postfix = convert_to_postfix(infix)
    print(postfix)

    command_array = postfix_to_command_array(postfix)
    print(command_array)

    output_graph = AGraph()
    output_graph.command_array = np.array(command_array, dtype=int)
    output_graph.set_local_optimization_params([2.0])
    output_graph.set_local_optimization_params([])
    print(output_graph)
