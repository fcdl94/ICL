import visdom
import numpy as np

EPS = 10e-6


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
        print(f"Epoch {epoch + 1:3d} : Train Loss {train_loss:.6f}, Train Acc {train_acc:.2f}\n"
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

        cn = conf.astype('float') / (conf.sum(axis=1)[:, np.newaxis] + EPS)

        self.vis.env = self.name + "_CF"
        self.vis.heatmap(cn)
        self.vis.env = self.name + "_TR"

        np.savetxt(self.path + f"/{self.name}CF{self.iteration}.csv", conf, delimiter=",")
        return conf
