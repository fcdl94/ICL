import torchvision
import numpy as np
import torch
from .abstract_dataset import IAbstractDataset
from torch.utils.data import DataLoader, Sampler, Subset, Dataset
from torchvision.datasets.folder import DatasetFolder
from .common import DatasetPrototypes


def get_index_of_classes(target, classes):
    l = []

    if isinstance(classes, int):  # if only one class is given, make it a list
        classes = [classes]

    for cl in classes:
        l.append(torch.nonzero(target == cl).squeeze())
    return torch.cat(l)


class IDADataset(IAbstractDataset):

    # implemento i.l. come N classi + M + M + M etc.
    # e ovviamente D(b0) != D(bi) con i>0
    def __init__(self, target, source, num_cl_first, num_cl_after, order_file=None,
                 run=0, batch_size=64, workers=1):
        super().__init__()

        assert isinstance(target, DatasetFolder), "target must be torchvision.DataFolder"
        assert isinstance(source, DatasetFolder), "source must be torchvision.DataFolder"

        # get important variables from dataset
        self.source = source
        self.target = target

        self.classes = target.classes
        self.num_classes = len(target.classes)
        self.class_to_idx = target.class_to_idx

        self.y_target = torch.tensor(target.targets)
        self.y_source = torch.tensor(source.targets)

        # get variable for this loader
        self.num_cl_first = num_cl_first  # N
        self.num_cl_after = num_cl_after  # M

        assert (self.num_classes - self.num_cl_first) % num_cl_after == 0, \
            "num_cl_after + N*num_cl_after must match the number of classes"
        self.num_iteration_max = 1 + (self.num_classes - num_cl_first) // num_cl_after

        # get parameters for the loader
        self.batch_size = batch_size
        self.workers = workers

        # get the order for the classes
        if order_file is None:
            self.full_order = [np.arange(self.num_classes)]  # if not specified go from zero to num_classes in order
        else:
            self.full_order = np.load(order_file)

        # init parameters
        self.iteration = 0
        self.order = torch.tensor(self.full_order[run])
        self.data_loader = None

    @property
    def order(self):
        return self.__order

    @order.setter
    def order(self, order):
        self.__order = order

    def get_X_of_class(self, idx, source=False):
        # this can ask too much memory! be careful in using
        idx_in_order = np.where(self.order == idx)[0]

        if idx_in_order >= self.num_cl_first:
            dataset = self.source
            target = self.y_source
        else:
            dataset = self.target
            target = self.y_target

        images = [dataset[x.item()][0] for x in get_index_of_classes(target, idx)]

        return torch.cat(images)

    def offset(self, iteration):
        return self.num_cl_first + self.num_cl_after*iteration

    def next_iteration(self, x_additional=None, y_additional=None, iteration=None):
        # iteration goes from 0 to max_iter-1
        # and the first iteration is a special one, where we return target and not source dataset with N classes (not M)

        if iteration is not None:
            self.iteration = iteration
        else:
            iteration = self.iteration
            self.iteration += 1

        if iteration == 0:
            # set up the right DataLoader
            dataset_full = self.target
            classes = self.order[0: self.num_cl_first]
            indices = get_index_of_classes(self.y_target, classes)

        elif 0 < iteration < self.num_iteration_max:
            dataset_full = self.source
            classes = self.order[self.offset(iteration-1): self.offset(iteration)]
            indices = get_index_of_classes(self.y_source, classes)

        else:
            raise Exception("You should stop before, you asked too many iterations")

        dataset = Subset(dataset_full, indices)  # here they are transformed with the transform defined in instantiation

        if x_additional is not None and y_additional is not None:
            # use the same transform of the dataset to augment the prototypes
            dataset_prototypes = DatasetPrototypes(x_additional, y_additional, dataset_full.transform)
            dataset += dataset_prototypes

        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers)

        return data_loader

    def reset_iteration(self):
        self.iteration = 0

    def test_dataloader(self, iteration=None, batch_size=None):
        if iteration is None:
            iteration = self.iteration-1
        if batch_size is None:
            batch_size = self.batch_size

        classes = self.order[0: self.offset(iteration)]
        print(classes)
        dataset_full = self.target
        indices = get_index_of_classes(self.y_target, classes)
        dataset = Subset(dataset_full, indices)

        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=self.workers)
        return data_loader
