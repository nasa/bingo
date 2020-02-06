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
IS_ARITY_2_MAP = {0: False,
                  1: False,
                  2: True,
                  3: True,
                  4: True,
                  5: True,
                  6: False,
                  7: False,
                  8: False,
                  9: False,
                  10: True,
                  11: False,
                  12: False}
IS_TERMINAL_MAP = {0: True,
                   1: True,
                   2: False,
                   3: False,
                   4: False,
                   5: False,
                   6: False,
                   7: False,
                   8: False,
                   9: False,
                   10: False,
                   11: False,
                   12: False}
OPERATOR_NAMES = {0: ["load", "x"],
                  1: ["constant", "c"],
                  2: ["add", "addition", "+"],
                  3: ["subtract", "subtraction", "-"],
                  4: ["multiply", "multiplication", "*"],
                  5: ["divide", "division", "/"],
                  6: ["sine", "sin"],
                  7: ["cosine", "cos"],
                  8: ["exponential", "exp", "e"],
                  9: ["logarithm", "log"],
                  10: ["power", "pow", "^"],
                  11: ["absolute value", "||", "|"],
                  12: ["square root", "sqrt"]}
