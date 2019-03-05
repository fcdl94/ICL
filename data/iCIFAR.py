import torchvision
import numpy as np
import torch
from .abstract_dataset import IAbstractDataset
import os

class ICIFAR(IAbstractDataset):

    def __init__(self, root, batch_size, nc_per_iter, order_file=None, download=True, run_number=0):
        super().__init__()
        # Load the dataset
        self.train_dataset = torchvision.datasets.CIFAR100(root=root, train=True, download=download)
        self.valid_dataset = torchvision.datasets.CIFAR100(root=root, train=False, download=download)
        # normalize data
        X_train_total = self.train_dataset.train_data.transpose(0, 3, 1, 2) / np.float32(255)  # shape n,3,32,32
        X_valid_total = self.valid_dataset.test_data.transpose(0, 3, 1, 2) / np.float32(255)
        pixel_means = np.mean(X_train_total, axis=0).reshape(1, 32, 32, 3).transpose(0, 3, 1, 2)
        self.pixel_means = pixel_means

        self.X_train = X_train_total - pixel_means
        self.X_valid = X_valid_total - pixel_means
        self.Y_valid = np.array(self.valid_dataset.test_labels)
        self.Y_train = np.array(self.train_dataset.train_labels)

        # get the order for incremental cifar
        if order_file is None:
            order_file = os.path.join(root, 'fixed_order.npy')
        self.full_order = np.load(order_file)

        # Init parameters that will really be initialized in set_run
        self.iteration = 0
        self.run = -1
        self.X_train_idx_per_class = None
        self.X_valid_idx_per_class = None
        self.set_run(run_number)

        # set additional parameters
        self.nb_cl = nc_per_iter
        self.batch_size = batch_size

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

    def set_run(self, run):
        self.run = run
        self.order = self.full_order[run]

        self.X_train_idx_per_class = self._unpack_data(self.X_train, self.Y_train, 500)
        self.X_valid_idx_per_class = self._unpack_data(self.X_valid, self.Y_valid, 100)
        # reset iteration counter
        self.iteration = 0

    def get_X_of_class(self, idx):
        return self.X_train[np.where(self.Y_train == idx)[0]]

    def _unpack_data(self, x, y, size):
        x_ = np.zeros(
            (100, size), dtype=np.float32)
        for i in range(100):
            x_[i, :] = np.where(y == self.order[i])[0]
        return x_.astype(int)

    def reset_iteration(self):
        self.iteration = 0

    def next_iteration(self, X_protoset=None, y_protoset=None, iteration=None):
        '''
        This function returns the data on which perform the epochs for the selected iteration.
        Training data are shuffled.

        :param X_protoset: Data to add at the dataset
        :param y_protoset: Data to add at the dataset
        :param iteration:  Selection for the iteration
        :return: X_train, y_train, X_valid, y_valid
        '''

        assert self.run >= 0, "You have not set up the run. Call set_run() before using this function."

        if iteration:
            self.iteration = iteration
        else:
            self.iteration += 1

        train_idx = np.concatenate(
            self.X_train_idx_per_class[(self.iteration - 1) * self.nb_cl:self.iteration * self.nb_cl])
        self.X_train_to_iter = self.X_train[train_idx]
        self.y_train_to_iter = self.Y_train[train_idx]

        valid_idx = np.concatenate(
            self.X_valid_idx_per_class[(self.iteration - 1) * self.nb_cl:self.iteration * self.nb_cl])
        self.X_valid_to_iter = self.X_valid[valid_idx]
        self.y_valid_to_iter = self.Y_valid[valid_idx]

        if X_protoset is not None and y_protoset is not None:
            # make the combined training set from exemplar and current samples
            self.X_train_to_iter = np.concatenate((self.X_train_to_iter, X_protoset), axis=0)
            self.y_train_to_iter = np.concatenate((self.y_train_to_iter, y_protoset))

        # Shuffle training data
        train_indices = np.arange(len(self.X_train_to_iter))
        np.random.shuffle(train_indices)
        self.X_train_to_iter = self.X_train_to_iter[train_indices, :, :, :]
        self.y_train_to_iter = self.y_train_to_iter[train_indices]

        self.X_train_to_iter = torch.tensor(self.X_train_to_iter)
        self.y_train_to_iter = torch.tensor(self.y_train_to_iter)
        self.X_valid_to_iter = torch.tensor(self.X_valid_to_iter)
        self.y_valid_to_iter = torch.tensor(self.y_valid_to_iter)

        return self.X_train_to_iter, self.y_train_to_iter, self.X_valid_to_iter, self.y_valid_to_iter

    # sarebbe piu' comodo farne un dataloader custom
    def minibatches(self, train=True, augment=False):
        if train:
            X = self.X_train_to_iter
            y = self.y_train_to_iter
        else:
            X = self.X_valid_to_iter
            y = self.y_valid_to_iter

        X = X.numpy()

        for start_idx in range(0, len(y) - self.batch_size + 1, self.batch_size):
            excerpt = slice(start_idx, start_idx + self.batch_size)
            if augment:
                # as in paper : 
                # pad feature arrays with 4 pixels on each side
                # and do random cropping of 32x32
                padded = np.pad(X[excerpt], ((0, 0), (0, 0), (4, 4), (4, 4)), mode='constant')
                random_cropped = np.zeros(X[excerpt].shape, dtype=np.float32)
                crops = np.random.random_integers(0, high=8, size=(self.batch_size, 2))
                for r in range(self.batch_size):
                    # Cropping and possible flipping
                    if np.random.randint(2) > 0:
                        random_cropped[r, :, :, :] = padded[r, :, crops[r, 0]:(crops[r, 0] + 32),
                                                     crops[r, 1]:(crops[r, 1] + 32)]
                    else:
                        random_cropped[r, :, :, :] = padded[r, :, crops[r, 0]:(crops[r, 0] + 32),
                                                     crops[r, 1]:(crops[r, 1] + 32)][:, :, ::-1]
                inp_exc = random_cropped
            else:
                inp_exc = X[excerpt]

            yield torch.tensor(inp_exc), y[excerpt]

    def minibatches_for_test(self, iteration, batch_size=None):

        if batch_size is None:
            batch_size = self.batch_size

        valid_idx = np.concatenate(
            self.X_valid_idx_per_class[0 : iteration * self.nb_cl])
        X = self.X_valid[valid_idx]
        y = self.Y_valid[valid_idx]

        for start_idx in range(0, len(y) - batch_size + 1, batch_size):
            excerpt = slice(start_idx, start_idx + batch_size)

            yield torch.tensor(X[excerpt]), torch.tensor(y[excerpt])
