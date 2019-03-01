from .icarl import ICarl


def get_dataset(name):
    if name.lower() == 'ICarl'.lower():
        return ICarl

    assert True, f"There is no methods called {name}."
