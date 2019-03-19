from .icarl import ICarl
import json
from .fine_tuning import FineTuning

methods = ["icarl", "lwf", "finetuning"]


def __parse_config__(config):

    return {"file": config}


def get_method(name, config=None, **kwargs):

    pars = {}

    if config is not None:
        print("Using config file: " + config)
        with open(config, "r") as read_file:
            pars = json.load(read_file)

    for i in kwargs:
        pars[i] = kwargs[i]

    if name.lower() == 'icarl':
        return ICarl(**pars)
    if name.lower() == 'lwf':
        return ICarl(**pars, mem_size=0)
    if name.lower() == 'finetuning':
        return FineTuning(**pars)

    assert True, f"There is no methods called {name}."
