import torchvision
import numpy as np
import torch
from .abstract_dataset import IAbstractDataset
from torch.utils.data import DataLoader, Sampler, Subset, Dataset
from torchvision.datasets.folder import DatasetFolder
from PIL import Image


def get_index_of_classes(target, classes):
    l = []

    if isinstance(classes, int):  # if only one class is given, make it a list
        classes = [classes]

    for cl in classes:
        l.append(torch.nonzero(target == cl).squeeze())
    return torch.cat(l)


class ClassSampler(Sampler):
    def __init__(self, target, classes):
        super().__init__()
        self.indices = get_index_of_classes(target, classes)

    def __iter__(self):
        return (self.indices[i].item() for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class DatasetPrototypes(Dataset):

    def __init__(self, x, y, transform=None):
        super().__init__()
        assert len(x) == len(y), "Error, the size of x and y must match"
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, index):

        sample = Image.fromarray(self.x[index])
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, self.y[index]

    def __len__(self):
        return len(self.y)


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
        if source:
            dataset = self.source
            target = self.y_source
        else:
            dataset = self.target
            target = self.y_target

        indices = [x.item for x in get_index_of_classes(target, idx)]
        return dataset[indices]

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
            dataset_prototypes = DatasetPrototypes(x_additional, y_additional, dataset_full.transform)
            dataset += dataset_prototypes

        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        return data_loader

    def reset_iteration(self):
        self.iteration = 0

    def minibatches(self, train=True, augment=True):
        assert self.iteration > 0, "You must call next_iteration before minibatches"

    def minibatches_for_test(self, iteration, batch_size=None):
        pass
