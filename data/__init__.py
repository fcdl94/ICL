dataset_names = {
    'ICIFAR': '.iCIFAR'}


def get_dataset(name):
    module = __import__(dataset_names[name])
    dataset_class = getattr(module, name)
    return dataset_class
