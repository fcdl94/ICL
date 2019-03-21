from .common import *
from datetime import datetime
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn

LR = 0.01
DECAY = 0.0001
EPOCHS = 70
LR_FACTOR = 5.

DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'


class FineTuning(AbstractMethod):

    def __init__(self, network, n_classes, nb_base, nb_incr,
                 log="FT", name="FT", epochs=EPOCHS, factor=LR_FACTOR,
                 lr_init=LR, decay=DECAY, device=DEVICE, **trash):

        super().__init__(network, n_classes, nb_base, nb_incr, log, name)

        self.lr = lr_init
        self.decay = decay
        self.momentum = 0.9
        self.device = device
        self.loss = nn.CrossEntropyLoss()
        self.epochs = epochs
        self.factor = 1/factor

        self.network.to(device)

    def fit(self, dataset, checkpoint=None, epochs=None):
        if epochs is not None:
            self.epochs = epochs

        start_iter = 0
        cumulative_accuracies = []
        self.dataset = dataset

        if checkpoint is not None:
            check_dict = torch.load(checkpoint)
            start_iter = check_dict['iteration']
            self.network.load_state_dict(check_dict['network'])

        for iteration in range(start_iter, self.iteration_total):

            # Prepare the training data for the current batch of classes
            train_loader, valid_loader = dataset.next_iteration()

            # TRAIN THIS ITERATION #
            print('Batch of classes number {0} arrives ...'.format(iteration + 1))

            # train for N epochs (after each epoch validate)
            self.incremental_fit(iteration, train_loader, valid_loader)

            # END OF TRAINING FOR THIS ITERATION #

            # Save training checkpoint
            # torch.save({
            #    'iteration': iteration,
            #    'network': self.network.state_dict(),
            # }, "checkpoint/iter_" + str(iteration) + "_checkpoint.pth.tar")

            # COMPUTE ACCURACY ##
            acc_cum = self.test(iteration)
            acc_new = self.test(iteration, cumulative=False)
            acc_base = self.test(0)

            print_accuracy(["FT"], acc_base, acc_new, acc_cum)

            cumulative_accuracies.append(acc_cum)

            i = 0
            name = 'FT'
            save_results(f"{self.log_folder}/{name}.csv", acc_base[i], acc_new[i], acc_cum[i])

            print("")

        acc_cum = []
        tot = self.iteration_total - 1

        for i in range(tot+1):
            acc_cum.append(self.test(i, cumulative=False))

        save_per_batch_result(f"{self.log_folder}/per-batch.csv", ["FT"], acc_cum)

        torch.save(
            {
                "network": self.network.state_dict()
            },
            f"models/FT{datetime.now().isoformat()}.pth"
            )

        return cumulative_accuracies

    def incremental_fit(self, iteration, train_loader, valid_loader):
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.network.parameters()),
                              lr=self.lr, momentum=self.momentum, weight_decay=self.decay)
        scheduler = MultiStepLR(optimizer, [int(0.7*self.epochs), int(0.9*self.epochs)], self.factor)

        for epoch in range(self.epochs):
            self.network.train()
            scheduler.step()
            train_loss = 0
            train_correct = 0
            train_total = 0
            valid_loss = 0
            valid_correct = 0
            valid_total = 0

            # In each epoch, we do a full pass over the training data:
            for inputs, targets in train_loader:

                optimizer.zero_grad()
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.network.forward(inputs)  # feature vector only
                prediction = self.network.predict(outputs)  # make the prediction

                loss_bx = self.loss(prediction, targets)  # CE loss
                loss_bx.backward()
                optimizer.step()

                train_loss += loss_bx.item()
                _, predicted = prediction.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()

            train_acc = 100. * train_correct / train_total

            self.network.eval()
            # In each epoch, we do a full pass over the training data:
            for inputs, targets in valid_loader:

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.network.forward(inputs)  # feature vector only
                prediction = self.network.predict(outputs)  # make the prediction

                loss_bx = self.loss(prediction, targets)  # CE loss

                valid_loss += loss_bx.item()
                _, predicted = prediction.max(1)
                valid_total += targets.size(0)
                valid_correct += predicted.eq(targets).sum().item()

            valid_acc = 100. * valid_correct / valid_total

            self.logger.log_training(epoch, train_loss/len(train_loader), train_acc,
                                     valid_loss/len(valid_loader), valid_acc, iteration)

    def predict(self, inputs):
        inputs = inputs.to(self.device)
        # compute prediction
        outputs = self.network.forward(inputs)  # returns embeddings
        prediction = self.network.predict(outputs).cpu().detach().numpy()  # return score classes as logits

        return prediction.cpu().detach()

    def test(self, iteration, cumulative=True, conf_matrix=False):
        data_loader = self.dataset.test_dataloader(iteration, cumulative=cumulative)
        correct = 0
        total = 0

        tot_target = []
        tot_pred = []

        for inputs, targets in data_loader:
            inputs = inputs.to(self.device)

            # compute prediction
            outputs = self.network.forward(inputs)  # returns embeddings
            prediction = self.network.predict(outputs).cpu().detach()  # return score classes as logits

            _, predicted = prediction.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if conf_matrix:
                tot_target += [i.item() for i in prediction]
                tot_pred += [i.item() for i in targets]

        if conf_matrix:
            self.logger.confusion_matrix(self.reorder_target(tot_target),
                                         self.reorder_target(tot_pred),
                                         self.n_classes)
        return [100. * correct / total]
