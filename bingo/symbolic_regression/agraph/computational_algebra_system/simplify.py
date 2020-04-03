from .constant_folding import fold_constants
from .automatic_simplification import automatic_simplify
from .interpreter import build_cas_expression, build_agraph_stack


def simplify(agraph_stack):
    print("starting stack: ", agraph_stack)

    cas_expression = build_cas_expression(agraph_stack)
    print("cas expression: ", cas_expression)

    cas_expression = automatic_simplify(cas_expression)
    print("simplified expression: ", cas_expression)

    cas_expression = fold_constants(cas_expression)
    print("simplified with folding: ", cas_expression)

    interpreted_stack = build_agraph_stack(cas_expression)
    print("ending stack: ", interpreted_stack)
    return interpreted_stack
