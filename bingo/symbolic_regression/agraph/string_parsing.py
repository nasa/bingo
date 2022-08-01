"""Tools for parsing strings into AGraphs"""
import re
import numpy as np

from bingo.symbolic_regression.agraph.operator_definitions \
    import INTEGER, VARIABLE, CONSTANT, ADDITION, SUBTRACTION, MULTIPLICATION, \
           DIVISION, SIN, COS, SINH, COSH, EXPONENTIAL, LOGARITHM, POWER, ABS, \
           SQRT

operators = {"+", "-", "*", "/", "^"}
functions = {"sin", "cos", "sinh", "cosh", "exp", "log", "abs", "sqrt"}
precedence = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
operator_map = {"+": ADDITION, "-": SUBTRACTION, "*": MULTIPLICATION,
                "/": DIVISION, "^": POWER, "X": VARIABLE, "x": VARIABLE,
                "C": CONSTANT, "c": CONSTANT,
                "sin": SIN, "cos": COS, "sinh": SINH, "cosh": COSH,
                "exp": EXPONENTIAL, "log": LOGARITHM, "abs": ABS,
                "sqrt": SQRT}
# matches X_### and C_### (case-insensitive)
var_or_const_pattern = re.compile(r"([XC])_(\d+)", re.IGNORECASE)
int_pattern = re.compile(r"\d+")  # matches ###
non_unary_op_pattern = re.compile(r"([*/^()])")  # matches *, /, ^, (, or )
negative_pattern = re.compile(r"-([^\s\d])")  # matches -N where N = non-number


def infix_to_postfix(infix_tokens):
    """Converts a list of infix tokens into its corresponding
    list of postfix tokens (e.g. ["a", "+", "b"] -> ["a", "b", "+"])

    Based on the Dijkstra's Shunting-yard algorithm

    Parameters
    ----------
    infix_tokens : list of str
        A list of infix string tokens

    Returns
    -------
    postfix_tokens : list of str
        A list of postfix string tokens corresponding
        to the expression given by infix_tokens
    """
    stack = []  # index -1 = top (the data structure, not a command array)
    output = []
    for token in infix_tokens:
        if token in operators:
            while len(stack) > 0 and stack[-1] in operators and \
                (precedence[stack[-1]] > precedence[token] or
                 precedence[stack[-1]] == precedence[token] and token != "^"):
                output.append(stack.pop())
            stack.append(token)
        elif token == "(" or token in functions:
            stack.append(token)
        elif token == ")":
            while len(stack) > 0 and stack[-1] != "(":
                output.append(stack.pop())
            if len(stack) == 0 or stack.pop() != "(":  # get rid of "("
                raise RuntimeError("Mismatched parenthesis")
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
    """Converts a list of postfix tokens to its corresponding command array
    and list of constants

    Parameters
    ----------
    postfix_tokens : list of str
        A list of postfix string tokens

    Returns
    -------
    command_array, constants : Nx3 numpy array of int, list of numeric
        A command array and list of constants
        corresponding to the expression given by the postfix_tokens
    """
    stack = []  # index -1 = top (the data structure, not a command array)
    command_array = []
    i = 0
    command_to_i = {}
    constants = []
    n_constants = 0

    for token in postfix_tokens:
        if token in operators:
            operands = stack.pop(), stack.pop()
            command = [operator_map[token], operands[1], operands[0]]
        elif token in functions:
            operand = stack.pop()
            command = [operator_map[token], operand, operand]
        else:
            var_or_const = var_or_const_pattern.fullmatch(token)
            integer = int_pattern.fullmatch(token)
            if var_or_const:
                groups = var_or_const.groups()
                command = [operator_map[groups[0]], int(groups[1]),
                           int(groups[1])]
            elif integer:
                operand = int(token)
                command = [INTEGER, operand, operand]
            else:
                try:
                    command = [CONSTANT, n_constants, n_constants]

                    constant = float(token)
                    constants.append(constant)
                    n_constants += 1
                except ValueError as err:
                    raise RuntimeError(f"Unknown token {token}") from err
        if tuple(command) in command_to_i:
            stack.append(command_to_i[tuple(command)])
        else:
            command_to_i[tuple(command)] = i
            command_array.append(command)
            stack.append(i)
            i += 1

    if len(stack) > 1:
        raise RuntimeError("Error evaluating postfix expression")

    return np.array(command_array, dtype=int), constants


def eq_string_to_infix_tokens(eq_string):
    """Converts an equation string to infix_tokens

    Parameters
    ----------
    eq_string : str
        A string corresponding to an equation

    Returns
    -------
    infix_tokens : list of str
        A list of string tokens that correspond
        to the expression given by eq_string
    """
    if any(bad_token in eq_string for bad_token in ["zoo", "I", "oo",
                                                       "nan"]):
        raise RuntimeError("Cannot parse inf/complex")
    eq_string = eq_string.replace(")(", ")*(").replace("**", "^")

    eq_string = negative_pattern.sub(r"-1 * \1", eq_string)
    # replace -token with -1.0 * token if token != a number

    tokens = non_unary_op_pattern.sub(r" \1 ", eq_string).split(" ")
    tokens = [x.lower() for x in tokens if x != ""]
    return tokens


def eq_string_to_command_array_and_constants(eq_string):
    """Converts an equation string to its corresponding command
    array and list of constants

    Parameters
    ----------
    eq_string : str
        A string corresponding to an equation

    Returns
    -------
    command_array, constants : Nx3 numpy array of int, list of numeric
        A command array and list of constants
        corresponding to the expression given by eq_string
    """
    infix_tokens = eq_string_to_infix_tokens(eq_string)
    postfix_tokens = infix_to_postfix(infix_tokens)
    return postfix_to_command_array_and_constants(postfix_tokens)
