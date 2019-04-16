import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.folder import ImageFolder
from .common import DatasetPrototypes, Subset, get_index_of_classes, split_dataset
import torchvision.transforms
import os


class DoubleDataset(torch.utils.data.Dataset):

    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __getitem__(self, index):
        if index >= len(self.dataset1):
            data_1 = self.dataset1[index % len(self.dataset1)]
        else:
            data_1 = self.dataset1[index]
        if index >= len(self.dataset2):
            data_2 = self.dataset2[index % len(self.dataset2)]
        else:
            data_2 = self.dataset2[index]

        return data_1, data_2

    def __len__(self):
        return max(len(self.dataset1), len(self.dataset2))


class IDADataloader:

    def __init__(self, root, target=None, source=None, test=None,
                 n_base=10, n_incr=10,
                 augmentation=None, transform=None, validation_size=0.2,
                 order_file=None, batch_size=64, run_number=0, workers=1):
        super().__init__()
        # Incremental domain adaptation: N classes + k_times*M classes
        # where Domain(N) != Domain(M) (if M==N we recover ICL without DA)

        assert transform is not None, "You should pass a transform to transform Image into Tensor"

        target, source, test = self.make_datasets(root, target, source, test, transform)

        # Creating data indices for target training and validation splits:
        target_train_indices, target_val_indices = \
            split_dataset(len(target.targets), True, validation_split=validation_size,
                          batch_size=batch_size)

        # Creating data indices for source training and validation splits:
        source_train_indices, source_val_indices = \
            split_dataset(len(source.targets), True, validation_split=validation_size,
                          batch_size=batch_size)

        # make training and validation dataset following the split
        self.target_train = Subset(target, target_train_indices, None)     # returns images as PIL Image
        self.target_valid = Subset(target, target_val_indices, transform)  # return tensors
        self.source_train = Subset(source, source_train_indices, None)     # returns images as PIL Image
        self.source_valid = Subset(source, source_val_indices, transform)  # return tensors

        self.test = test  # return tensors

        # catch training labels to compute the images of only some classes
        self.target_train_labels = torch.tensor([target.targets[i] for i in target_train_indices])
        self.target_valid_labels = torch.tensor([target.targets[i] for i in target_val_indices])
        self.source_train_labels = torch.tensor([source.targets[i] for i in source_train_indices])
        self.source_valid_labels = torch.tensor([source.targets[i] for i in source_val_indices])

        self.test_labels = torch.tensor(test.targets)

        # store transform and augmentation function
        self.augmentation = augmentation  # this works as data augmentation
        self.transform = transform  # this works as ToTensor, without changing the images.

        # store num of classes
        self.num_classes = len(target.classes)

        # check the number of classes for incremental class learning (base + k*incr)
        self.num_cl_first = n_base  # N
        self.num_cl_after = n_incr  # k*M

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
        self.data_loader = None

    def make_datasets(self, root, target, source, test, transform):
        target = ImageFolder(os.path.join(root, target), None, None)
        source = ImageFolder(os.path.join(root, source), None, None)
        test = ImageFolder(os.path.join(root, test), transform)
        return target, source, test

    @property
    def order(self):
        return self.__order

    @order.setter
    def order(self, order):
        self.__order = order

    def get_images_of_class(self, idx):
        # this can ask too much memory! be careful in using

        # select the right dataset
        idx_in_order = np.where(self.order == idx)[0]

        if idx_in_order >= self.num_cl_first:
            dataset = self.source_train
            target = self.source_train_labels
        else:
            dataset = self.target_train
            target = self.target_train_labels

        # dataset[x.item()][0] is PIL Image (I'm not using any transform for training datasets)
        images = [dataset[x.item()][0] for x in get_index_of_classes(target, [idx])]

        return images  # this is a list of PIL images

    def get_dataloader_of_class(self, idx, custom_transform=None):
        # select the right dataset
        idx_in_order = np.where(self.order == idx)[0]

        if idx_in_order >= self.num_cl_first:
            dataset = self.source_train
            target = self.source_train_labels
        else:
            dataset = self.target_train
            target = self.target_train_labels

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

    def next_iteration(self, target_proto=None, source_proto=None, iteration=None):

        if iteration is not None:
            self.iteration = iteration
        else:
            iteration = self.iteration
            self.iteration += 1

        if self.augmentation is not None:
            transform = torchvision.transforms.Compose([self.augmentation, self.transform])
        else:
            transform = self.transform

        if iteration == 0:  # we are in base classes
            dataset_full = self.target_train
            classes = self.order[0: self.num_cl_first]
            indices = get_index_of_classes(self.target_train_labels, classes)
            train_dataset = Subset(dataset_full, indices, transform)

            valid_indices = get_index_of_classes(self.target_valid_labels, classes)
            valid_dataset = Subset(self.target_valid, valid_indices)

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers)
            valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False,num_workers=self.workers)

        elif 0 < iteration < self.num_iteration_max:  # we are in the source classes, return the concat
            dataset_full = self.source_train
            classes = self.order[self.offset(iteration-1): self.offset(iteration)]
            indices = get_index_of_classes(self.source_train_labels, classes)
            train_dataset = Subset(dataset_full, indices, transform)

            valid_indices = get_index_of_classes(self.source_valid_labels, classes)
            valid_dataset = Subset(self.source_valid, valid_indices)

            if source_proto is not None:  # we should be in iteration > 1
                # use the same transform of the dataset to augment the prototypes
                dataset_prototypes = DatasetPrototypes(*source_proto, transform)
                train_dataset += dataset_prototypes  # combine train and prototype source dataset T^i + P^i

            if target_proto is not None:  # we should be in iteration >= 1
                # use the same transform of the dataset to augment the prototypes
                target_prototypes = DatasetPrototypes(*target_proto, transform)  # make protos dataset
                target_dataset = target_prototypes
                train_dataset = DoubleDataset(train_dataset, target_dataset)  # make double ds to return both tuples
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size // 2, shuffle=True,
                                          num_workers=self.workers)
            else:  # we are not using target prototypes even if we are in iteration >= 1
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                          num_workers=self.workers)

            valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False,
                                      num_workers=self.workers)
        else:
            raise Exception("You should stop before, you asked too many iterations")

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

        if len(classes) == 0:  # handle the case where there are no base classes
            dataset = Subset(dataset_full, [])
        else:
            indices = get_index_of_classes(self.test_labels, classes)
            dataset = Subset(dataset_full, indices)

        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=self.workers)
        return data_loader


class CifarDataloader(IDADataloader):
    def make_datasets(self, root, target, source, test, transform):
        # FOR CIFAR! Load the dataset
        target = torchvision.datasets.CIFAR100(root=root, train=True, download=False, transform=None)
        source = target
        test = torchvision.datasets.CIFAR100(root=root, train=False, download=False, transform=transform)
        return target, source, test


class SingleDataloader(IDADataloader):
    def make_datasets(self, root, target, source, test, transform):
        target = ImageFolder(os.path.join(root, target), None, None)
        source = target
        test = ImageFolder(os.path.join(root, test), transform)
        return target, source, test


class MNIST_to_SVHN_Dataloader(IDADataloader):
    def make_datasets(self, root, target, source, test, transform):
        target = torchvision.datasets.MNIST(root, train=True, transform=torchvision.transforms.Grayscale(3))
        source = torchvision.datasets.SVHN(root)
        source.targets = source.labels
        test = torchvision.datasets.MNIST(root, train=False, transform=torchvision.transforms.Compose(
                                          [torchvision.transforms.Grayscale(3), transform]))
        return target, source, test


class MNISTDataloader(IDADataloader):
    def make_datasets(self, root, target, source, test, transform):
        target = torchvision.datasets.MNIST(root, train=True, transform=torchvision.transforms.Grayscale(3))
        source = target
        test = torchvision.datasets.MNIST(root, train=False, transform=torchvision.transforms.Compose(
                                              [torchvision.transforms.Grayscale(3), transform]))

        test.targets = test.targets.numpy()  # only to remove the warning
        return target, source, test
