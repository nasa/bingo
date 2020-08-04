
def test_cpp_agraph():
    try:
        from bingocpp import AGraph
        bingocpp = True
    except ModuleNotFoundError:
        bingocpp = False

    if not bingocpp:
        raise ModuleNotFoundError("Bingocpp AGraph could not be loaded."
                                  " Its tests will be skipped.")


def test_cpp_evaluation_backend():
    try:
        from bingocpp import evaluation_backend
        bingocpp = True
    except ModuleNotFoundError:
        bingocpp = False

    if not bingocpp:
        raise ModuleNotFoundError("Bingocpp evaluation_backend could not be "
                                  "loaded."
                                  " Its tests will be skipped.")


def test_cpp_simplification_backend():
    try:
        from bingocpp import simplification_backend
        bingocpp = True
    except ModuleNotFoundError:
        bingocpp = False

    if not bingocpp:
        raise ModuleNotFoundError("Bingocpp simplification_backend could not "
                                  "be loaded."
                                  " Its tests will be skipped.")