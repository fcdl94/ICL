import os
import numpy as np
import torch
from .abstract import AbstractIncrementalDataloader
from torch.utils.data import DataLoader
from .common import DatasetPrototypes, Subset, get_index_of_classes, split_dataset
import torchvision.transforms
from torchvision.datasets import ImageFolder


class IncrementalDataloader(AbstractIncrementalDataloader):

    def __init__(self, root, target,
                 n_base=10, n_incr=10,
                 augmentation=None, transform=None, validation_size=0.2,
                 order_file=None, batch_size=128, run_number=0, workers=1):
        super().__init__()
        # Incremental domain adaptation: N classes + M classes + M + M etc.
        # where Domain(N) != Domain(M)
        assert transform is not None, "You should pass a transform to transform Image into Tensor"

        target = ImageFolder(os.path.join(root, target), None, None)

        # Creating data indices for training and validation splits:

        train_indices, val_indices = \
            split_dataset(len(target), True, validation_split=validation_size,
                          batch_size=batch_size)

        # Creating PT data samplers
        self.train = Subset(target, train_indices)
        self.valid = Subset(target, val_indices, transform)
        self.test = self.valid

        self.augmentation = augmentation  # this works as data augmentation
        self.transform = transform  # this works as ToTensor, without changing the images.

        self.classes = target.classes
        self.num_classes = len(target.classes)

        self.train_labels = torch.tensor([target.targets[i] for i in train_indices])
        self.valid_labels = torch.tensor([target.targets[i] for i in val_indices])
        self.test_labels = self.valid_labels

        # get variable for this loader
        self.num_cl_first = n_base  # N
        self.num_cl_after = n_incr  # M

        if n_incr == 0:
            n_incr = 1
        assert (self.num_classes - self.num_cl_first) % n_incr == 0, \
            "num_cl_after + N*num_cl_after must match the number of classes"
        self.num_iteration_max = 1 + (self.num_classes - n_base) // n_incr

        # get parameters for the loader
        self.batch_size = batch_size
        self.workers = workers

        # get the order for the classes
        if order_file is None:
            self.full_order = [np.arange(self.num_classes)]  # if not specified go from zero to num_classes in order
            run_number = 0
        else:
            self.full_order = np.genfromtxt(os.path.join(root, order_file), delimiter=",").astype(int)

        # init parameters
        self.iteration = 0
        self.order = self.full_order[run_number]

    @property
    def order(self):
        return self.__order

    @order.setter
    def order(self, order):
        self.__order = order

    def get_images_of_class(self, idx):
        # this can ask too much memory! be careful in using

        dataset = self.train
        target = self.train_labels

        # dataset[x.item()][0] is PIL Image (I'm not using any transform)
        images = [dataset[x.item()][0] for x in get_index_of_classes(target, [idx])]

        return images  # this is a list of PIL images

    def get_dataloader_of_class(self, idx, custom_transform=None):
        # select the right dataset
        dataset = self.train
        target = self.train_labels

        indices = get_index_of_classes(target, [idx])

        if custom_transform is None:
            transform = self.transform
        else:
            transform = torchvision.transforms.Compose([custom_transform, self.transform])

        dataset = Subset(dataset, indices, transform)

        sampler = torch.utils.data.SequentialSampler(dataset)  # to guarantee sequentiality of indices

        return DataLoader(dataset, batch_size=self.batch_size, sampler=sampler, num_workers=self.workers)

    def offset(self, iteration):
        if iteration == -1:
            return 0
        if iteration >= self.num_iteration_max:
            iteration = self.num_iteration_max - 1
        return self.num_cl_first + self.num_cl_after*iteration

    def next_iteration(self, x_additional=None, y_additional=None, iteration=None):

        # iteration goes from 0 to max_iter-1
        # and the first iteration is a special one,
        # where we return target and not source dataset with N classes (not M)

        if iteration is not None:
            self.iteration = iteration
        else:
            iteration = self.iteration
            self.iteration += 1

        # chose the dataset
        if iteration >= self.num_iteration_max:
            raise Exception("You should stop before, you asked too many iterations")

        if self.augmentation is not None:
            transform = torchvision.transforms.Compose([self.augmentation, self.transform])
        else:
            transform = self.transform

        classes = self.order[self.offset(iteration-1): self.offset(iteration)]
        train_indices = get_index_of_classes(self.train_labels, classes)

        valid_indices = get_index_of_classes(self.valid_labels, classes)

        train_dataset = Subset(self.train, train_indices, transform)
        valid_dataset = Subset(self.valid, valid_indices)

        if x_additional is not None and y_additional is not None:
            # use the same transform of the dataset to augment the prototypes
            dataset_prototypes = DatasetPrototypes(x_additional, y_additional, transform)
            train_dataset += dataset_prototypes

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.workers)

        return train_loader, valid_loader

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

        dataset_full = self.test
        indices = get_index_of_classes(self.test_labels, classes)
        dataset = Subset(dataset_full, indices)

        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=self.workers)
        return data_loader
