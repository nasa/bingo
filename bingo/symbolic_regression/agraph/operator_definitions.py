"""
Attributes
----------
IS_ARITY_2_MAP : dict {int: bool}
                 A map of node number to boolean that states whether the
                 node has arity 2 (as opposed to 1)
IS_TERMINAL_MAP : dict {int: bool}
                 A map of node number to boolean that states whether the
                 node is a terminal
OPERATOR_NAMES : dict{int: list(string)}
                 A map of node number to common names for the node
"""
INTEGER = -1
VARIABLE = 0
CONSTANT = 1
ADDITION = 2
SUBTRACTION = 3
MULTIPLICATION = 4
DIVISION = 5
SIN = 6
COS = 7
SINH = 8
COSH = 9
EXPONENTIAL = 10
LOGARITHM = 11
POWER = 12
ABS = 13
SQRT = 14
IS_ARITY_2_MAP = {INTEGER: False,
                  VARIABLE: False,
                  CONSTANT: False,
                  ADDITION: True,
                  SUBTRACTION: True,
                  MULTIPLICATION: True,
                  DIVISION: True,
                  SIN: False,
                  COS: False,
                  SINH: False,
                  COSH: False,
                  EXPONENTIAL: False,
                  LOGARITHM: False,
                  POWER: True,
                  ABS: False,
                  SQRT: False}
IS_TERMINAL_MAP = {INTEGER: True,
                   VARIABLE: True,
                   CONSTANT: True,
                   ADDITION: False,
                   SUBTRACTION: False,
                   MULTIPLICATION: False,
                   DIVISION: False,
                   SIN: False,
                   COS: False,
                   SINH: False,
                   COSH: False,
                   EXPONENTIAL: False,
                   LOGARITHM: False,
                   POWER: False,
                   ABS: False,
                   SQRT: False}
OPERATOR_NAMES = {INTEGER: ["integer"],
                  VARIABLE: ["load", "x"],
                  CONSTANT: ["constant", "c"],
                  ADDITION: ["add", "addition", "+"],
                  SUBTRACTION: ["subtract", "subtraction", "-"],
                  MULTIPLICATION: ["multiply", "multiplication", "*"],
                  DIVISION: ["divide", "division", "/"],
                  SIN: ["sine", "sin"],
                  COS: ["cosine", "cos"],
                  SINH: ["sineh", "sinh"],
                  COSH: ["cosineh", "cosh"],
                  EXPONENTIAL: ["exponential", "exp", "e"],
                  LOGARITHM: ["logarithm", "log"],
                  POWER: ["power", "pow", "^"],
                  ABS: ["absolute value", "||", "|"],
                  SQRT: ["square root", "sqrt"]}
