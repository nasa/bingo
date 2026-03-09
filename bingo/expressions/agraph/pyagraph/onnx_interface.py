"""ONNX model generation for AGraph expressions.

Converts a command stack (with constants and integers) into an ONNX
model that can be used for deployment and inference.
"""

import numpy as np
from .operators import (
    VARIABLE,
    CONSTANT,
    INTEGER,
    ADDITION,
    SUBTRACTION,
    MULTIPLICATION,
    DIVISION,
    POWER,
    SAFE_POWER,
    SQUARE,
    CUBE,
    SQRT,
    ABS,
    EXPONENTIAL,
    LOGARITHM,
    SIN,
    COS,
    TAN,
    ARCSIN,
    ARCCOS,
    ARCTAN,
    SINH,
    COSH,
    TANH,
    ARITY_2_IDS,
)

try:
    from onnx import numpy_helper, TensorProto
    from onnx.helper import (
        make_model,
        make_node,
        make_graph,
        make_tensor_value_info,
    )
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


ONNX_FUNCTIONS = {
    ADDITION: "Add",
    SUBTRACTION: "Sub",
    MULTIPLICATION: "Mul",
    DIVISION: "Div",
    SIN: "Sin",
    COS: "Cos",
    TAN: "Tan",
    EXPONENTIAL: "Exp",
    LOGARITHM: "Log",
    POWER: "Pow",
    ABS: "Abs",
    SQRT: "Sqrt",
    SAFE_POWER: "Pow",
    SINH: "Sinh",
    COSH: "Cosh",
    TANH: "Tanh",
    ARCSIN: "Asin",
    ARCCOS: "Acos",
    ARCTAN: "Atan",
}

ABS_SAFETY = {LOGARITHM, SQRT, SAFE_POWER}


def make_onnx_model(command_array, constants, integers,
                    name="bingo_expression"):
    """Create an ONNX model from a command stack.

    Parameters
    ----------
    command_array : Nx3 array of int
        The command stack.
    constants : tuple of numeric
        Numeric constants in the equation.
    integers : tuple of int
        Integer values in the equation.
    name : str, optional
        Name for the ONNX model. Default ``"bingo_expression"``.

    Returns
    -------
    onnx.ModelProto
        ONNX model.

    Raises
    ------
    ImportError
        If the ``onnx`` package is not installed.
    """
    if not ONNX_AVAILABLE:
        raise ImportError(
            "The onnx package is required. Install it with: pip install onnx"
        )

    nodes = []
    slice_inds = set()

    input_ = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
    output = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
    initializer = numpy_helper.from_array(
        np.array(constants, dtype=np.float32), name="C"
    )
    nodes.append(make_node("Constant", [], ["ax0"], value_ints=[0]))
    nodes.append(make_node("Constant", [], ["ax1"], value_ints=[1]))

    for i, (op, p1, p2) in enumerate(command_array):
        output_name = f"O{i}" if i < len(command_array) - 1 else "Y"

        if op == INTEGER:
            val = float(integers[p1]) if p1 < len(integers) else 0.0
            nodes.append(
                make_node("Constant", [], [output_name], value_float=val)
            )
        elif op == VARIABLE:
            for p in [p1, p1 + 1]:
                if p not in slice_inds:
                    slice_inds.add(p)
                    nodes.append(
                        make_node("Constant", [], [f"s{p}"], value_ints=[p])
                    )
            nodes.append(
                make_node("Slice", ["X", f"s{p1}", f"s{p1+1}", "ax1"],
                          [output_name])
            )
        elif op == CONSTANT:
            for p in [p1, p1 + 1]:
                if p not in slice_inds:
                    slice_inds.add(p)
                    nodes.append(
                        make_node("Constant", [], [f"s{p}"], value_ints=[p])
                    )
            nodes.append(
                make_node("Slice", ["C", f"s{p1}", f"s{p1+1}", "ax0"],
                          [output_name])
            )
        elif op in ABS_SAFETY:
            nodes.append(
                make_node("Abs", [f"O{p1}"], [f"{output_name}aux"])
            )
            inps = (
                [f"{output_name}aux", f"O{p2}"]
                if op in ARITY_2_IDS
                else [f"{output_name}aux"]
            )
            nodes.append(make_node(ONNX_FUNCTIONS[op], inps, [output_name]))
        elif op == SQUARE:
            nodes.append(
                make_node("Mul", [f"O{p1}", f"O{p1}"], [output_name])
            )
        elif op == CUBE:
            nodes.append(
                make_node("Mul", [f"O{p1}", f"O{p1}"],
                          [f"{output_name}aux"])
            )
            nodes.append(
                make_node("Mul", [f"{output_name}aux", f"O{p1}"],
                          [output_name])
            )
        else:
            inps = (
                [f"O{p1}", f"O{p2}"]
                if op in ARITY_2_IDS
                else [f"O{p1}"]
            )
            nodes.append(make_node(ONNX_FUNCTIONS[op], inps, [output_name]))

    return make_model(
        make_graph(nodes, name, [input_], [output], [initializer])
    )
