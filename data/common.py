from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage
import torch


def get_index_of_classes(target, classes):
    l = []

    if isinstance(classes, int):  # if only one class is given, make it a list
        classes = [classes]

    for cl in classes:
        l.append(torch.nonzero(target == cl).squeeze())
    return torch.cat(l)


class DatasetPrototypes(Dataset):

    def __init__(self, x, y, transform=None):
        """

        :param x: PIL Images
        :param y: integer list
        :param transform: torchvision.transforms that can transform PIL into tensor
        """
        super().__init__()
        assert len(x) == len(y), "Error, the size of x and y must match"
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, index):

        sample = self.x[index]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, int(self.y[index])

    def __len__(self):
        return len(self.y)


class Subset(Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        sample, target = self.dataset[self.indices[idx]]
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return len(self.indices)
