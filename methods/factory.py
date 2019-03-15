from .icarl import ICarl

methods = ["icarl", "lwf"]


def get_method(name, config=None, **kwargs):
    if config is not None:
        print("Using config file: " + config)
    if name.lower() == 'ICarl'.lower():
        return ICarl(**kwargs)
    if name.lower() == 'lwf':
        return ICarl(**kwargs, mem_size=0)

    assert True, f"There is no methods called {name}."
