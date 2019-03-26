import torchvision
import numpy as np
import torch
from .abstract import AbstractIncrementalDataloader
import os
from .common import DatasetPrototypes, Subset, get_index_of_classes, split_dataset
from torch.utils.data import DataLoader


class ICIFAR(AbstractIncrementalDataloader):

    MEAN = [0.5071, 0.4867, 0.4408]
    STD = [0.2675, 0.2565, 0.2761]

    def __init__(self, root, download=True,
                 n_base=10, n_incr=10,
                 augmentation=None, transform=None, validation_size=0.,
                 order_file=None, batch_size=128, run_number=0, workers=1):

        super().__init__()

        # Load the dataset
        train_dataset = \
            torchvision.datasets.CIFAR100(root=root, train=True, download=download, transform=None)
        self.test_dataset = \
            torchvision.datasets.CIFAR100(root=root, train=False, download=download, transform=transform)

        self.test_target = torch.tensor(self.test_dataset.targets)  # this does not work with torchvision < 0.2.2

        if validation_size == 0:
            self.train_dataset = train_dataset
            self.valid_dataset = self.test_dataset
            self.train_target = torch.tensor(train_dataset.targets)  # this does not work with torchvision < 0.2.2
            self.valid_target = self.test_target
        else:
            # Creating data indices for training and validation splits
            train_indices, val_indices = split_dataset(len(train_dataset.targets), True,
                                                       validation_split=validation_size, batch_size=batch_size)

            # make training and validation dataset following the split
            self.train_dataset = Subset(train_dataset, train_indices, None)
            self.valid_dataset = Subset(train_dataset, val_indices, transform)

            # get targets to compute indices
            self.train_target = torch.tensor([train_dataset.targets[i] for i in train_indices])
            self.valid_target = torch.tensor([train_dataset.targets[i] for i in val_indices])

        if augmentation is not None:
            self.augmentation = torchvision.transforms.Compose([augmentation, transform])
        else:
            self.augmentation = transform
        self.transform = transform  # this works as ToTensor, without changing the image content.

        # get the order for incremental cifar
        if order_file is None:
            order_file = os.path.join(root, 'cifar-100-python', 'cifar_order.npy')
        self.full_order = np.load(order_file)
        self.order = self.full_order[run_number]

        # Init parameters that will really be initialized in set_run
        self.iteration = 0

        # set additional parameters
        self.num_cl_first = n_base
        self.num_cl_after = n_incr

        if n_incr == 0:
            n_incr = 1

        assert (100 - self.num_cl_first) % n_incr == 0, \
            "num_cl_after + N*num_cl_after must match the number of classes"
        self.num_iteration_max = 1 + (100 - n_incr) // n_incr

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

        # dataset[x.item()][0] is PIL Image (I'm not using any transform)
        images = [dataset[x.item()][0] for x in get_index_of_classes(target, [idx])]

        return images  # this is a list of PIL images

    def get_dataloader_of_class(self, idx, custom_transform=None):
        indices = get_index_of_classes(self.train_target, [idx])

        if custom_transform is None:
            transform = self.transform
        else:
            transform = torchvision.transforms.Compose([custom_transform, self.transform])

        dataset = Subset(self.train_dataset, indices, transform)

        sampler = torch.utils.data.SequentialSampler(dataset)  # to guarantee sequentiality of indices

        return DataLoader(dataset, batch_size=self.batch_size, sampler=sampler, num_workers=self.workers)

    def offset(self, iteration):
        if iteration == -1:
            return 0
        return self.num_cl_first + self.num_cl_after*iteration

    def next_iteration(self, x_additional=None, y_additional=None, iteration=None):

        if iteration is not None:
            self.iteration = iteration
        else:
            iteration = self.iteration
            self.iteration += 1

        if iteration > self.num_iteration_max:
            raise Exception("You should stop before, you asked too many iterations")

        classes = self.order[self.offset(iteration-1): self.offset(iteration)]
        indices = get_index_of_classes(self.train_target, classes)

        train_dataset = Subset(self.train_dataset, indices, self.augmentation)

        if x_additional is not None and y_additional is not None:
            # use the same transform of the dataset to augment the prototypes
            dataset_prototypes = DatasetPrototypes(x_additional, y_additional, self.augmentation)
            train_dataset += dataset_prototypes

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers)

        # Make validation of current class
        valid_indices = get_index_of_classes(self.valid_target, classes)

        valid_dataset = Subset(self.valid_dataset, valid_indices)

        val_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.workers)

        return train_loader, val_loader

    def test_dataloader(self, iteration=None, cumulative=True, batch_size=None):
        if iteration is None:
            iteration = self.iteration-1
        if batch_size is None:
            batch_size = self.batch_size

        if cumulative:
            start_offset = 0
        else:
            start_offset = self.offset(iteration-1)

        if iteration > self.num_iteration_max:
            raise Exception("You should stop before, you asked too many iterations")

        classes = self.order[start_offset: self.offset(iteration)]

        dataset_full = self.test_dataset
        indices = get_index_of_classes(self.test_target, classes)
        dataset = Subset(dataset_full, indices)

        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=self.workers)
        return data_loader
