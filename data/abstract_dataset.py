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
        # questa funzione non mi piace, dovremmo pensare di sostituirla (serve solo ad ICaRL per tenere prototipi)
        # ICaRL dovrebbe gestirsi il suo meccanismo di memoria che stora le immagini quando gli arrivano
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
