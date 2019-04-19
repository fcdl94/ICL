import visdom
import numpy as np
import logging
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt


class TensorboardXLogger:
    def __init__(self, path, name):
        self.writer = SummaryWriter(log_dir=path)
        self.iteration = 0

    def log_training(self, epoch, train_loss, train_acc, valid_loss, valid_acc, iteration, **kwargs):
        logging.info(f"Epoch {epoch + 1:3d} : Train Loss {train_loss:.6f}, Train Acc {train_acc:.2f}\n"
                     f"          : Valid Loss {valid_loss:.6f}, Valid Acc {valid_acc:.2f}")
        if self.iteration != iteration:
            self.iteration = iteration

        self.writer.add_scalar('train_loss', train_loss, epoch)
        self.writer.add_scalar('train_acc', train_acc, epoch)
        self.writer.add_scalar('valid_loss', valid_loss, epoch)
        self.writer.add_scalar('valid_acc', valid_acc, epoch)

        for k, v in kwargs:
            self.writer.add_scalar(k, v, epoch)

    def conf_matrix_figure(self, cm, classes):
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=f'Confusion Matrix {self.iteration}',
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return fig

    def confusion_matrix(self, y, y_hat, n_classes):
        conf = np.zeros((n_classes, n_classes))

        for i in range(len(y)):
            conf[y[i], y_hat[i]] += 1

        cm = conf.astype('float') / (conf.sum(axis=1)+0.000001)[:, np.newaxis]

        fig = self.conf_matrix_figure(cm, np.arange(n_classes))
        self.writer.add_figure('conf_matrix', fig, self.iteration)

        avg_acc = np.diag(cm).mean() * 100.
        self.writer.add_scalar('avg_acc', avg_acc, self.iteration)
        logging.info(f"Per class accuracy: {avg_acc}")
        return conf


class VisdomLogger:
    def __init__(self, path, name):
        self.vis = visdom.Visdom()
        self.vis.env = name + "_TR"
        self.name = name
        self.path = path
        self.iteration = 0
        self.reset_stat()

    def reset_stat(self):
        self.epoch = []
        self.train_loss = []
        self.train_acc = []
        self.valid_loss = []
        self.valid_acc = []

    def log_training(self, epoch, train_loss, train_acc, valid_loss, valid_acc, iteration=0):
        logging.info(f"Epoch {epoch + 1:3d} : Train Loss {train_loss:.6f}, Train Acc {train_acc:.2f}\n"
                     f"          : Valid Loss {valid_loss:.6f}, Valid Acc {valid_acc:.2f}")

        if self.iteration != iteration:
            self.reset_stat()
            self.iteration = iteration

        self.epoch.append(epoch+1)
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        self.valid_loss.append(valid_loss)
        self.valid_acc.append(valid_acc)
        self.print_log()

    def print_log(self):
        base = self.iteration * 4
        self.vis.line(
            X=self.epoch,
            Y=self.train_loss,
            opts={
                'title': f'Training Loss {self.iteration}'
            },
            name='Training Loss',
            win=base
        )
        self.vis.line(
            X=self.epoch,
            Y=self.valid_loss,
            opts={
                'title': f'Validation Loss {self.iteration}'
            },
            name='Validation Loss',
            win=base+1
        )
        self.vis.line(
            X=self.epoch,
            Y=self.train_acc,
            opts={
                'title': f'Training Accuracy {self.iteration}'
            },
            name='Training Accuracy',
            win=base+2
        )
        self.vis.line(
            X=self.epoch,
            Y=self.valid_acc,
            opts={
                'title': f'Validation Accuracy {self.iteration}'
            },
            name='Validation Accuracy',
            win=base+3
        )

    def confusion_matrix(self, y, y_hat, n_classes):
        conf = np.zeros((n_classes, n_classes))

        for i in range(len(y)):
            conf[y[i], y_hat[i]] += 1

        cm = conf.astype('float') / (conf.sum(axis=1)+0.000001)[:, np.newaxis]

        self.vis.env = self.name + "_CF"
        self.vis.heatmap(cm, opts={
            'title': f'Confusion Matrix {self.iteration}',
            'xlabel': 'Pred Class',
            'ylabel': 'Real Class'}
        )
        self.vis.env = self.name + "_TR"

        np.savetxt(self.path + f"/{self.name}CF{self.iteration}.csv", conf, delimiter=",")
        logging.info(f"Per class accuracy: { np.diag(cm).mean() * 100.}")
        return conf
