from abc import ABC, abstractmethod


class AbstractIncrementalDataloader(ABC):

    def __init__(self):
        super().__init__()
        self.order = None
        self.iteration = 0

    def reset_iteration(self):
        self.iteration = 0

    @property
    def order(self):
        raise NotImplementedError

    @order.setter
    def order(self, order):
        raise NotImplementedError

    @abstractmethod
    def get_images_of_class(self, idx):
        # it must return the Images as Tensors (without any transformation)
        raise NotImplementedError

    @abstractmethod
    def get_dataloader_of_class(self, idx):
        # it must return a DataLoader that returns the images not augmented!
        raise NotImplementedError

    @abstractmethod
    def next_iteration(self, x_additional=None, y_additional=None, iteration=None):
        # it must return a DataLoader
        raise NotImplementedError

    @abstractmethod
    def test_dataloader(self, iteration=None, batch_size=None):
        # it must return a DataLoader for cumulative
        raise NotImplementedError
