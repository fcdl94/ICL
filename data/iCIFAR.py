import torchvision
import numpy as np
import torch
from .abstract import AbstractIncrementalDataloader
import os
from .common import DatasetPrototypes, Subset
from torch.utils.data import DataLoader


def get_index_of_classes(target, classes):
    l = []

    if isinstance(classes, int):  # if only one class is given, make it a list
        classes = [classes]

    for cl in classes:
        l.append(torch.nonzero(target == cl).squeeze())
    return torch.cat(l)


class ICIFAR(AbstractIncrementalDataloader):

    MEAN = [0.5071, 0.4867, 0.4408]
    STD = [0.2675, 0.2565, 0.2761]

    def __init__(self, root, download=True,
                 num_cl_first=10, num_cl_after=10,
                 augmentation=None, transform=None,
                 order_file=None, batch_size=64, run_number=0, workers=1):
        super().__init__()
        # Load the dataset
        self.train_dataset = \
            torchvision.datasets.CIFAR100(root=root, train=True, download=download, transform=None)
        self.valid_dataset = \
            torchvision.datasets.CIFAR100(root=root, train=False, download=download, transform=transform)

        self.augmentation = torchvision.transforms.Compose([augmentation, transform])  # this works as data augmentation
        self.transform = transform  # this works as ToTensor, without changing the images.

        # get targets to compute indices
        self.train_target = torch.tensor(self.train_dataset.targets)  # this does not work with torchvision < 0.2.2
        self.valid_target = torch.tensor(self.valid_dataset.targets)  # this does not work with torchvision < 0.2.2

        # get the order for incremental cifar
        if order_file is None:
            order_file = os.path.join(root, 'cifar-100-python', 'cifar_order.npy')
        self.full_order = np.load(order_file)
        self.order = self.full_order[run_number]

        # Init parameters that will really be initialized in set_run
        self.iteration = 0

        # set additional parameters
        self.num_cl_first = num_cl_first
        self.num_cl_after = num_cl_after

        assert (100 - self.num_cl_first) % num_cl_after == 0, \
            "num_cl_after + N*num_cl_after must match the number of classes"
        self.num_iteration_max = 1 + (100 - num_cl_first) // num_cl_after

        # set additional parameters
        self.batch_size = batch_size
        self.workers = workers

        # init parameters for iteration
        self.X_train_to_iter = None
        self.y_train_to_iter = None

        self.X_valid_to_iter = None
        self.y_valid_to_iter = None

    @property
    def order(self):
        return self.__order

    @order.setter
    def order(self, order):
        self.__order = order

    def get_images_of_class(self, idx):
        # this function can ask too much memory! Use it only with small dataset!
        dataset = self.train_dataset
        target = self.train_target

        trans = torchvision.transforms.ToTensor()

        # dataset[x.item()][0] is PIL Image (I'm not using any transform)
        images = [trans(dataset[x.item()][0]).unsqueeze(0) for x in get_index_of_classes(target, [idx])]

        return torch.cat(images)

    def get_dataloader_of_class(self, idx):
        indices = get_index_of_classes(self.train_target, [idx])
        dataset = Subset(self.train_dataset, indices, self.transform)

        sampler = torch.utils.data.SequentialSampler(dataset)

        return DataLoader(dataset, batch_size=self.batch_size, sampler=sampler, num_workers=self.workers)

    def offset(self, iteration):
        if iteration == -1:
            return 0
        return self.num_cl_first + self.num_cl_after*iteration

    def next_iteration(self, x_additional=None, y_additional=None, iteration=None):
        '''
        This function returns the data on which perform the epochs for the selected iteration.
        Training data are shuffled.

        :param x_additional: Data to add at the dataset
        :param y_additional: Data to add at the dataset
        :param iteration:  Selection for the iteration
        :return: DataLoader to iterate across data
        '''

        if iteration is not None:
            self.iteration = iteration
        else:
            iteration = self.iteration
            self.iteration += 1

        if iteration > self.num_iteration_max:
            raise Exception("You should stop before, you asked too many iterations")

        dataset_full = self.train_dataset
        classes = self.order[self.offset(iteration-1): self.offset(iteration)]
        indices = get_index_of_classes(self.train_target, classes)

        dataset = Subset(dataset_full, indices, self.augmentation)

        if x_additional is not None and y_additional is not None:
            # use the same transform of the dataset to augment the prototypes
            dataset_prototypes = DatasetPrototypes(x_additional, y_additional, self.augmentation)
            dataset += dataset_prototypes

        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers)

        return data_loader

    def test_dataloader(self, iteration=None, batch_size=None):
        if iteration is None:
            iteration = self.iteration-1
        if batch_size is None:
            batch_size = self.batch_size

        if iteration > self.num_iteration_max:
            raise Exception("You should stop before, you asked too many iterations")

        classes = self.order[0: self.offset(iteration)]

        dataset_full = self.valid_dataset
        indices = get_index_of_classes(self.valid_target, classes)
        dataset = Subset(dataset_full, indices)

        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=self.workers)
        return data_loader
