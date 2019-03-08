from abc import ABC, abstractmethod


class IAbstractDataset(ABC):

    def __init__(self):
        super().__init__()
        self.order = None

    @property
    def order(self):
        raise NotImplementedError

    @order.setter
    def order(self, order):
        raise NotImplementedError

    @abstractmethod
    def get_X_of_class(self, idx):
        # it must return a torch.tensor
        raise NotImplementedError

    @abstractmethod
    def next_iteration(self, X_protoset=None, y_protoset=None, iteration=None):
        # it must return a DataLoader
        raise NotImplementedError

    @abstractmethod
    def reset_iteration(self):
        raise NotImplementedError

    @abstractmethod
    def minibatches(self, train=True, augment=True):
        raise NotImplementedError

    @abstractmethod
    def minibatches_for_test(self, iteration, batch_size=None):
        raise NotImplementedError
