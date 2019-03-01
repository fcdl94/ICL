methods_names = {
    'ICarl': '.icarl'}


def get_method(name):
    module = __import__(methods_names[name])
    method_class = getattr(module, name)
    return method_class
