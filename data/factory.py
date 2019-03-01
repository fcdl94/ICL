from .iCIFAR import ICIFAR


def get_dataset(name):
    if name.lower() == 'ICIFAR'.lower():
        return ICIFAR

    assert True, f"There is no dataset called {name}."
