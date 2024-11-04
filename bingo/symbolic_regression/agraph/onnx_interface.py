"""Wrapper around ONNX for deploying bingo equations

This module is a wrapper around the onnx package that allows for conversion of
bingo AGraph objects to the onnx standard.
"""

import numpy as np
from onnx import numpy_helper, TensorProto
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_tensor_value_info,
)

from .operator_definitions import (
    INTEGER,
    VARIABLE,
    CONSTANT,
    ADDITION,
    SUBTRACTION,
    MULTIPLICATION,
    DIVISION,
    SIN,
    COS,
    EXPONENTIAL,
    LOGARITHM,
    POWER,
    ABS,
    SQRT,
    SAFE_POWER,
    SINH,
    COSH,
    IS_ARITY_2_MAP,
)

ONNX_FUNCTIONS = {
    ADDITION: "Add",
    SUBTRACTION: "Sub",
    MULTIPLICATION: "Mul",
    DIVISION: "Div",
    SIN: "Sin",
    COS: "Cos",
    EXPONENTIAL: "Exp",
    LOGARITHM: "Log",
    POWER: "Pow",
    ABS: "Abs",
    SQRT: "Sqrt",
    SAFE_POWER: "Pow",
    SINH: "Sinh",
    COSH: "Cosh",
}

ABS_SAFETY = set([LOGARITHM, SQRT, SAFE_POWER])


def make_onnx_model(command_array, constants, name="bingo_equation"):
    """creates an onnx model from bingo agraph represntation

    Parameters
    ----------
    command_array : Nx3 array of int
        acyclic graph stack
    constants : tuple of numeric
        numeric constants that are used in the equation
    name : str, optional
        name for onnx model, by default "bingo_equation"

    Returns
    -------
    onnx Model
        onnx represntation of the AGraph
    """

    nodes = []
    slice_inds = set()

    input_ = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
    output = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
    initializer = numpy_helper.from_array(np.array(constants), name="C")
    nodes.append(make_node("Constant", [], ["ax0"], value_ints=[0]))
    nodes.append(make_node("Constant", [], ["ax1"], value_ints=[1]))

    for i, (op, p1, p2) in enumerate(command_array):
        output_name = f"O{i}" if i < len(command_array) - 1 else "Y"
        if op == INTEGER:
            nodes.append(
                make_node("Constant", [], [output_name], value_float=float(p1))
            )
        elif op == VARIABLE:
            for p in [p1, p1 + 1]:
                if p not in slice_inds:
                    slice_inds.add(p)
                    nodes.append(make_node("Constant", [], [f"s{p}"], value_ints=[p]))
            nodes.append(
                make_node("Slice", ["X", f"s{p1}", f"s{p1+1}", "ax1"], [output_name])
            )
        elif op == CONSTANT:
            for p in [p1, p1 + 1]:
                if p not in slice_inds:
                    slice_inds.add(p)
                    nodes.append(make_node("Constant", [], [f"s{p}"], value_ints=[p]))
            nodes.append(
                make_node("Slice", ["C", f"s{p1}", f"s{p1+1}", "ax0"], [output_name])
            )
        elif op in ABS_SAFETY:
            nodes.append(make_node("Abs", [f"O{p1}"], [f"{output_name}aux"]))
            inps = (
                [f"{output_name}aux", f"O{p2}"]
                if IS_ARITY_2_MAP[op]
                else [f"{output_name}aux"]
            )
            nodes.append(make_node(ONNX_FUNCTIONS[op], inps, [output_name]))
        else:
            inps = [f"O{p1}", f"O{p2}"] if IS_ARITY_2_MAP[op] else [f"O{p1}"]
            nodes.append(make_node(ONNX_FUNCTIONS[op], inps, [output_name]))

    return make_model(make_graph(nodes, name, [input_], [output], [initializer]))
