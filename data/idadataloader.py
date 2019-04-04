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

    def __init__(self, root, target, source,
                 n_base=10, n_incr=10,
                 augmentation=None, transform=None, validation_size=0.2,
                 order_file=None, batch_size=64, run_number=0, workers=1):
        super().__init__()
        # Incremental domain adaptation: N classes + M classes + M + M etc.
        # where Domain(N) != Domain(M)

        assert transform is not None, "You should pass a transform to transform Image into Tensor"

        target = ImageFolder(os.path.join(root, target), None, None)
        source = ImageFolder(os.path.join(root, source), None, None)

        # Creating data indices for training and validation splits:
        train_indices, val_indices = \
            split_dataset(len(target.targets), True, validation_split=validation_size,
                          batch_size=batch_size)

        # make training and validation dataset following the split
        self.train = Subset(target, train_indices, None)     # returns images as PIL Image
        self.valid = Subset(target, val_indices, transform)  # return tensors
        self.test = self.valid                               # return tensors
        self.source = source                                 # returns images as PIL Image

        self.train_labels = torch.tensor([target.targets[i] for i in train_indices])
        self.valid_labels = torch.tensor([target.targets[i] for i in val_indices])
        self.test_labels = self.valid_labels
        self.source_labels = torch.tensor(source.targets)

        self.augmentation = augmentation  # this works as data augmentation
        self.transform = transform  # this works as ToTensor, without changing the images.

        self.num_classes = len(target.classes)

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
        self.data_loader = None

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
            dataset = self.source
            target = self.source_labels
        else:
            dataset = self.train
            target = self.train_labels

        # dataset[x.item()][0] is PIL Image (I'm not using any transform)
        images = [dataset[x.item()][0] for x in get_index_of_classes(target, [idx])]

        return images  # this is a list of PIL images

    def get_dataloader_of_class(self, idx, custom_transform=None):
        # select the right dataset
        idx_in_order = np.where(self.order == idx)[0]

        if idx_in_order >= self.num_cl_first:
            dataset = self.source
            target = self.source_labels
        else:
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
            dataset_full = self.train
            classes = self.order[0: self.num_cl_first]
            indices = get_index_of_classes(self.train_labels, classes)
            train_dataset = Subset(dataset_full, indices, transform)

            valid_indices = get_index_of_classes(self.valid_labels, classes)
            valid_dataset = Subset(self.valid, valid_indices)

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers)
            valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False,
                                      num_workers=self.workers)

        elif 0 < iteration < self.num_iteration_max:  # we are in the source classes, return the concat
            dataset_full = self.source
            classes = self.order[self.offset(iteration-1): self.offset(iteration)]
            indices = get_index_of_classes(self.source_labels, classes)
            train_dataset = Subset(dataset_full, indices, transform)

            valid_indices = get_index_of_classes(self.valid_labels, classes)
            valid_dataset = Subset(self.valid, valid_indices)

            if source_proto is not None:
                # use the same transform of the dataset to augment the prototypes
                dataset_prototypes = DatasetPrototypes(*source_proto, transform)
                train_dataset += dataset_prototypes

            if target_proto is not None:
                # use the same transform of the dataset to augment the prototypes
                target_prototypes = DatasetPrototypes(*target_proto, transform)
                target_dataset = target_prototypes
                train_dataset = DoubleDataset(train_dataset, target_dataset)
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size // 2, shuffle=True,
                                          num_workers=self.workers)
            else:
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
