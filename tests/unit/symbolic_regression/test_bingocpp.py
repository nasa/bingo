def test_bingocpp_is_built():
    try:
        from bingocpp.build import symbolic_regression as bingocpp
    except ModuleNotFoundError:
        bingocpp = None

    if bingocpp is None:
        raise ModuleNotFoundError("Bingocpp could not be loaded. All bingocpp "
                                  "tests will be skipped.")