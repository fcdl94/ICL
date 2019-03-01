from abc import ABC, abstractmethod


class IAbstractDataset(ABC):

    def __init__(self):
        super().__init__()
        self.order = None

    @abstractmethod
    @property
    def order(self):
        pass

    @abstractmethod
    @order.setter
    def order(self, order):
        pass

    @abstractmethod
    def get_X_of_class(self, idx):
        pass

    @abstractmethod
    def next_iteration(self, X_protoset=None, y_protoset=None, iteration=None):
        pass

    @abstractmethod
    def reset_iteration(self):
        pass

    @abstractmethod
    def minibatches(self, train=True, augment=True):
        pass

    @abstractmethod
    def minibatches_for_test(self, iteration, batch_size=None):
        pass
