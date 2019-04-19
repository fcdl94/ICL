import os
from abc import ABC, abstractmethod
from .logger import TensorboardXLogger
import logging


def log_training(epoch, train_loss, train_acc, valid_loss, valid_acc):
    logging.debug(f"Epoch {epoch + 1:3d} : Train Loss {train_loss:.6f}, Train Acc {train_acc:.2f}\n"
                  f"          : Valid Loss {valid_loss:.6f}, Valid Acc {valid_acc:.2f}")


def save_results(file, acc_base, acc_new, acc_cum):
    print_header = False
    if not os.path.isfile(file):
        print_header = True
    with open(file, "a") as f:
        if print_header:
            f.write("acc_base,acc_new,acc_cum\n")
        f.write(f"{acc_base},{acc_new},{acc_cum}\n")


def save_per_batch_result(file, methods, acc):
    with open(file, "a") as f:
        f.write("Batch," + ",".join(methods) + "\n")

        for i in range(len(acc)):
            v = ",".join([f"{x:.3f}" for x in acc[i]])
            f.write(f"{i},{v}\n")


def create_log_folder(log):
    if not os.path.isdir(log):
        os.makedirs(log)


def print_accuracy(methods, acc_base, acc_new, acc_cum):
    logging.info("Cumulative results")
    for i, m in enumerate(methods):
        logging.info(f"  top 1 accuracy {m:<15}:\t{acc_cum[i]:.2f}")
    logging.info("New batch results")
    for i, m in enumerate(methods):
        logging.info(f"  top 1 accuracy {m:<15}:\t{acc_new[i]:.2f}")
    logging.info("First results")
    for i, m in enumerate(methods):
        logging.info(f"  top 1 accuracy {m:<15}:\t{acc_base[i]:.2f}")
    logging.info("")


class AbstractMethod(ABC):

    def __init__(self, network, n_classes, nb_base, nb_incr, log, name, lr_init, decay):
        self.network = network
        self.n_classes = n_classes
        self.nb_base = nb_base
        self.nb_incr = nb_incr
        if nb_incr == 0 and nb_base == self.n_classes:
            nb_incr = 1
        self.iteration_total = ((self.n_classes - nb_base) // nb_incr) + 1
        self.epochs = 0

        self.log_folder = log
        create_log_folder(log)
        self.logger = TensorboardXLogger(self.log_folder, name)

        logging.debug(f"Starting the method {name}")
        self.name = name
        self.dataset = None

    @abstractmethod
    def fit(self, dataset, checkpoint=None, epochs=None):
        pass

    @abstractmethod
    def incremental_fit(self, iteration, train_loader, valid_dataloader):
        pass

    @abstractmethod
    def test(self, iteration, cumulative=True,):
        pass

    def reorder_target(self, target):
        assert self.dataset is not None, "self.dataset is None! Please, set it up before call this method"
        m = {val: idx for idx, val in enumerate(self.dataset.order)}
        order_target = [m[v] for v in target]
        return order_target
