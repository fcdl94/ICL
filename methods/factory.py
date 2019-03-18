from .icarl import ICarl

methods = ["icarl", "lwf", "finetuning"]


def __parse_config__(config):

    return {"file": config}


def get_method(name, config=None, **kwargs):
    if config is not None:
        print("Using config file: " + config)

    if name.lower() == 'icarl':
        return ICarl(**kwargs)
    if name.lower() == 'lwf':
        return ICarl(**kwargs, mem_size=0)
    if name.lower() == 'finetuning':
        return ICarl(**kwargs, distillation=False)

    assert True, f"There is no methods called {name}."
