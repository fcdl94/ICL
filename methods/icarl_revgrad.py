from .icarl_da import ICarlDA
import torch
import torch.nn as nn
import numpy as np


class ICarlRG(ICarlDA):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.domain_criterion = nn.BCEWithLogitsLoss()
        self.lam = 0
        self.count = 0
        self.constant = 0.01

    def observe(self, epoch, iteration, train_loader, valid_loader, scheduler, optimizer):
        self.network.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        scheduler.step()

        # steps
        start_steps = epoch * len(train_loader)
        total_steps = self.epochs * len(train_loader)

        if iteration == 0 or not self.protos:  # if I DON'T use protos (I don't use them in first iteration as well)
            self.lam = 0
            self.network.set_target()
            for batch in train_loader:
                optimizer.zero_grad()

                loss_bx, trt_tot, trt_crc = self._compute_loss(batch, iteration)

                loss_bx.backward()
                optimizer.step()
                # update stats
                train_loss += loss_bx.item()
                train_total += trt_tot
                train_correct += trt_crc
        else:  # if I USE protos
            batch_idx = 0
            for source_loader, target_loader in train_loader:

                p = float(batch_idx + start_steps) / total_steps
                self.lam = 2. / (1. + np.exp(-10 * p)) - 1

                optimizer.zero_grad()

                # train the source
                self.network.set_source()
                loss_bx_src, tr_tot, tr_crc = self._compute_loss(source_loader, iteration, target=False)
                train_total += tr_tot
                train_correct += tr_crc

                # train the target
                self.network.set_target()
                loss_bx_tar, tr_tot, tr_crc = self._compute_loss(target_loader, iteration)
                train_total += tr_tot
                train_correct += tr_crc

                loss_bx = loss_bx_src + loss_bx_tar
                loss_bx.backward()
                optimizer.step()

                train_loss += loss_bx.item()
                batch_idx += 1

        # make validation
        self.network.eval()
        if iteration == 0:
            self.network.set_target()
        else:
            self.network.set_source()
        test_loss = 0
        test_correct = 0
        test_total = 0
        for inputs, targets_prep in valid_loader:
            targets = np.zeros((inputs.shape[0], self.n_classes), np.float32)
            targets[range(len(targets_prep)), targets_prep.type(torch.int32)] = 1.

            inputs = inputs.to(self.device)

            outputs = self.network.forward(inputs)  # make the embedding
            outputs = self.network.predict(outputs)  # make the prediction with sigmoid, making g_y(xi)
            targets = torch.tensor(targets).to(outputs.device)
            targets_prep = torch.LongTensor(targets_prep).to(outputs.device)

            loss_bx = self.loss(outputs, targets)  # without distillation? -> YES, validation only on new classes

            test_loss += loss_bx.item()
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets_prep).sum().item()

        # normalize and print stats
        train_acc = 100. * train_correct / train_total
        test_acc = 100. * test_correct / test_total

        return train_loss, train_acc, test_loss, test_acc

    def _compute_loss(self, loader, iteration, target=True):
        inputs, targets_prep = loader

        if target:
            domain = torch.ones(inputs.shape[0], 1).to(self.device)   # target is one
        else:
            domain = torch.zeros(inputs.shape[0], 1).to(self.device)  # source is zero

        targets = np.zeros((inputs.shape[0], self.n_classes), np.float32)
        targets[range(len(targets_prep)), targets_prep.type(torch.int32)] = 1.

        inputs = inputs.to(self.device)

        outputs = self.network.forward(inputs)  # feature vector only
        prediction = self.network.predict(outputs)  # make the prediction with sigmoid, making g_y(xi)
        domain_pred = self.network.discriminate_domain(outputs, self.lam) # the predicted domain

        targets = torch.tensor(targets).to(outputs.device)
        targets_prep = torch.LongTensor(targets_prep).to(outputs.device)

        if iteration > 0 and self.distillation:  # apply distillation
            outputs_old = self.network2.forward(inputs)
            prediction_old = self.network2.predict(outputs_old)
            to = self.compute_num_classes(iteration - 1)  # until the number of classes of last iteration
            targets[:, np.array(self.dataset.order[range(0, to)])] = \
                torch.sigmoid(prediction_old[:, np.array(self.dataset.order[range(0, to)])])

        loss_dm = self.domain_criterion(domain_pred, domain)
        loss_bx = self.loss(prediction, targets)  # joins classification and distillation losses
        _, predicted = prediction.max(1)
        train_total = targets.size(0)
        train_correct = predicted.eq(targets_prep).sum().item()

        total_loss = loss_bx + self.constant * loss_dm
        if self.count == 250:
            #print(f"Lam {self.lam} --- Class Loss {loss_bx:.4f} --- Domain Loss {self.lam * loss_dm}")
            self.count = 0
        self.count += 1

        return total_loss, train_total, train_correct
