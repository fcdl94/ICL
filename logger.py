#import visdom
import numpy as np
import logging
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt


class TensorboardXLogger:
    def __init__(self, path, name):
        self.writer = SummaryWriter(log_dir=path+"/"+name)
        self.name = name
        self.path = path

    def log_training(self, epoch, train_loss, train_acc, valid_loss, valid_acc, domain_loss, class_loss, **kwargs):
        logging.info(f"Epoch {epoch + 1:3d} : Train Loss {train_loss:.6f}, Train Acc {train_acc:.2f}\n"
                     f"          : Valid Loss {valid_loss:.6f}, Valid Acc {valid_acc:.2f}")

        self.writer.add_scalar(f'loss/train', train_loss, epoch)
        self.writer.add_scalar(f'loss/valid', train_loss, epoch)
        self.writer.add_scalar(f'snnl/class', class_loss, epoch)
        self.writer.add_scalar(f'snnl/domain', domain_loss, epoch)
        self.writer.add_scalar(f'acc/train', train_acc, epoch)
        self.writer.add_scalar(f'acc/val', valid_acc, epoch)

        for k in kwargs:
            self.writer.add_scalar(k, kwargs[k], epoch)

    def conf_matrix_figure(self, cm, classes):
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=f'Confusion Matrix',
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
        self.writer.add_figure('conf_matrix', fig, close=True)

        avg_acc = np.diag(cm).mean() * 100.
        print(f"Per class accuracy: {avg_acc}")
        return conf
