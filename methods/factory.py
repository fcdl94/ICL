from .icarl import ICarl

methods = ["icarl"]


def get_method(name):
    if name.lower() == 'ICarl'.lower():
        return ICarl

    assert True, f"There is no methods called {name}."
