from .icarl import ICarl
import json
from .fine_tuning import FineTuning

methods = ["icarl", "lwf", "finetuning"]


def __parse_config__(config):
    pars = {}

    if config is not None:
        print("Using config file: " + config)
        with open(config, "r") as read_file:
            pars = json.load(read_file)
    return pars


def get_method(m_name, config=None, **kwargs):

    pars = __parse_config__(config)

    for i in kwargs:
        pars[i] = kwargs[i]

    if m_name.lower() == 'icarl':
        return ICarl(**pars)
    if m_name.lower() == 'lwf':
        return ICarl(**pars, mem_size=0)
    if m_name.lower() == 'finetuning':
        return FineTuning(**pars)

    assert True, f"There is no methods called {m_name}."
