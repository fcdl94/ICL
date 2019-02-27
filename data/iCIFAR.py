import torchvision
import numpy as np


class ICIFAR:

    def __init__(self, root, batch_size, nc_per_iter, order_file, download=False, run_number=0):
        # Load the dataset
        self.train_dataset = torchvision.datasets.CIFAR100(root=root, train=True, download=download)
        self.valid_dataset = torchvision.datasets.CIFAR100(root=root, train=False, download=download)
        # normalize data
        X_train_total = self.train_dataset.train_data.transpose(0, 3, 1, 2) / np.float32(255)  # shape n,3,32,32
        X_valid_total = self.valid_dataset.test_data.transpose(0, 3, 1, 2) / np.float32(255)
        pixel_means = np.mean(X_train_total, axis=0).reshape(1, 32, 32, 3).transpose(0, 3, 1, 2)

        self.X_train = X_train_total - pixel_means
        self.X_valid = X_valid_total - pixel_means
        self.Y_valid = np.array(self.valid_dataset.test_labels)
        self.Y_train = np.array(self.train_dataset.train_labels)

        # get the order for incremental cifar
        self.full_order = np.load(order_file)

        # Init parameters that will really be initialized in set_run
        self.iteration = 0
        self.run = 0
        self.order = None
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

    def set_run(self, run):
        self.run = run
        self.order = self.full_order[run]

        self.X_train_idx_per_class = self._unpack_data(self.X_train, self.Y_train, 500)
        self.X_valid_idx_per_class = self._unpack_data(self.X_valid, self.Y_valid, 100)
        # reset iteration counter
        self.iteration = 0

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

        if X_protoset and y_protoset:
            # make the combined training set from exemplar and current samples
            self.X_train_to_iter = np.concatenate((self.X_train_to_iter, X_protoset), axis=0)
            self.y_train_to_iter = np.concatenate((self.y_train_to_iter, y_protoset))

        # Shuffle training data
        train_indices = np.arange(len(self.X_train_to_iter))
        np.random.shuffle(train_indices)
        self.X_train_to_iter = self.X_train_to_iter[train_indices, :, :, :]
        self.y_train_to_iter = self.y_train_to_iter[train_indices]

        return self.X_train_to_iter, self.y_train_to_iter, self.X_valid_to_iter, self.y_valid_to_iter

    # sarebbe piu' comodo farne un dataloader custom
    def minibatches(self, train=True, augment=False):
        if train:
            X = self.X_train_to_iter
            y = self.y_train_to_iter
        else:
            X = self.X_valid_to_iter
            y = self.y_valid_to_iter

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

            yield np.array(inp_exc, dtype=np.float32), y[excerpt]
