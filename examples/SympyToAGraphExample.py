import sympy as sy

from bingo.symbolic_regression.agraph.string_parsing import sympy_string_to_agraph

if __name__ == '__main__':
    sympy_string = str(sy.expand("(1./24.)*5e-5*X_0*(10.**3 - 2.*10.*X_0**2 + X_0**3)"))
    # NOTE that this is a string, sy.expand does not return a string!,
    # need to wrap str() around sy.expand to get a string

    # IMPORTANT notice how powers are integers (e.g. X_0**3 != X_0**3.0),
    # since there is no way I know of distinguishing between a const and int
    # other than decimal place, keep integers without decimal places if you
    # want them to be interpreted as integers instead of constants

    print("expanded sympy eq:", sympy_string)

    bingo_agraph = sympy_string_to_agraph(sympy_string)
    print("agraph from sympy string", bingo_agraph)

    bingo_agraph._needs_opt = True  # keeping with Geoff's standard, _needs_opt
    # is False by default, so if you want to run CLO, you need to set this
    # to True
