from abc import ABC, abstractmethod
from torch.utils.data import DataLoader


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
        # it must return the Images as PIL (without any transformation)
        raise NotImplementedError

    @abstractmethod
    def get_dataloader_of_class(self, idx, custom_transform=None):
        # it must return a DataLoader that returns the images not augmented!
        raise NotImplementedError

    @abstractmethod
    def next_iteration(self, x_additional=None, y_additional=None, iteration=None) -> DataLoader:
        """
        This function returns the data on which perform the epochs for the selected iteration.
        Training data are shuffled.

        :param x_additional: Data to add at the dataset
        :param y_additional: Data to add at the dataset
        :param iteration:  Selection for the iteration (0 for the base batch)
        :return: DataLoader to iterate across data
        """
        raise NotImplementedError

    @abstractmethod
    def test_dataloader(self, iteration=None, cumulative=True, batch_size=None)  -> DataLoader:
        # it must return a DataLoader for cumulative
        raise NotImplementedError
