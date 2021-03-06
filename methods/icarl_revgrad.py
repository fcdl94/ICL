from .icarl_da import ICarlDA
import torch
import torch.nn as nn
import numpy as np
import logging


class ICarlRG(ICarlDA):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.domain_criterion = nn.BCEWithLogitsLoss()
        self.lam = 0
        self.count = 0
        self.constant = 1

    def observe(self, epoch, iteration, train_loader, valid_loader, optimizer):
        self.network.train()
        self.network2.eval()

        train_loss = 0
        train_correct = 0
        train_total = 0
        self.count = 0
        # steps
        start_steps = epoch * len(train_loader)
        total_steps = self.epochs * len(train_loader)

        if iteration == 0 or not self.protos:  # if I DON'T use protos (I don't use them in first iteration as well)
            self.lam = 0
            const = self.constant
            self.constant = 0
            self.network.set_target()
            for batch in train_loader:
                optimizer.zero_grad()

                loss, trt_tot, trt_crc, loss_cl = self._compute_loss(batch, iteration)

                loss.backward()
                optimizer.step()
                # update stats
                train_loss += loss_cl.item()
                train_total += trt_tot
                train_correct += trt_crc
            self.constant = const
        else:  # if I USE protos
            batch_idx = 0
            for source_loader, target_loader in train_loader:

                p = float(batch_idx + start_steps) / total_steps
                self.lam = 2. / (1. + np.exp(-10 * p)) - 1

                optimizer.zero_grad()

                # train the source
                self.network.set_source()
                loss_bx_src, tr_tot, tr_crc, loss_cl = self._compute_loss(source_loader, iteration, target=False)
                train_total += tr_tot
                train_correct += tr_crc
                train_loss += loss_cl.item()

                # train the target
                self.network.set_target()
                loss_bx_tar, tr_tot, tr_crc, loss_cl = self._compute_loss(target_loader, iteration)
                train_total += tr_tot
                train_correct += tr_crc
                train_loss += loss_cl.item()

                loss_bx = loss_bx_src + loss_bx_tar
                loss_bx.backward()
                optimizer.step()

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

            logits, feats = self.network.forward(inputs)  # make the embedding
            outputs = self.network.predict(logits)  # make the prediction with sigmoid, making g_y(xi)
            targets = torch.tensor(targets).to(self.device)
            targets_prep = torch.LongTensor(targets_prep).to(self.device)

            loss_bx = self.loss(outputs, targets)  # without distillation? -> YES, validation only on new classes

            test_loss += loss_bx.item()
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets_prep).sum().item()

        # normalize and print stats
        train_acc = 100. * train_correct / train_total
        test_acc = 100. * test_correct / test_total

        test_loss /= len(valid_loader)
        train_loss /= len(train_loader)

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

        logits, feats = self.network.forward(inputs)  # make the embedding
        prediction = self.network.predict(logits)  # make the prediction with sigmoid, making g_y(xi)
        domain_pred = self.network.discriminate_domain(feats, self.lam) # the predicted domain

        targets = torch.tensor(targets).to(self.device)
        targets_prep = torch.LongTensor(targets_prep).to(self.device)

        if iteration > 0 and self.distillation:  # apply distillation
            logits_old, feat_old = self.network2.forward(inputs)
            prediction_old = self.network2.predict(logits_old)
            to = self.compute_num_classes(iteration - 1)  # until the number of classes of last iteration
            targets[:, np.array(self.dataset.order[range(0, to)])] = \
                torch.sigmoid(prediction_old[:, np.array(self.dataset.order[range(0, to)])])

        loss_dm = self.domain_criterion(domain_pred, domain)
        loss_bx = self.loss(prediction, targets)  # joins classification and distillation losses
        _, predicted = prediction.max(1)
        train_total = targets.size(0)
        train_correct = predicted.eq(targets_prep).sum().item()

        domain_acc = torch.sigmoid(domain_pred.detach()).mean().cpu().item()
        if not target:
            domain_acc = 1 - domain_acc

        total_loss = loss_bx + self.constant * loss_dm
        if self.count % 400 == 0 or self.count % 400 == 1:
            logging.info(f"{self.count:5d}: Lam {self.lam:.4f} --- Class Loss {loss_bx:.4f} "
                         f"--- Domain Loss {loss_dm:4f} --- {'TarDom' if target else 'SrcDom'} Acc {domain_acc:.3f}")
        self.count += 1

        return total_loss, train_total, train_correct, loss_bx
