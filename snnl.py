# -*- coding: utf-8 -*-
"""SNNL.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1e2cVRQnZFIyxndhJ-cjecweVdJmmPTLb
"""

import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

from data import MNISTM
from data import DoubleDataset
from data.common import get_index_of_classes
from networks.svhn import lenet_net

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('name', default="snnl_", help='The name of experiment')
parser.add_argument('-D', default=-1, type=float)
parser.add_argument('-Y', default=1, type=float)


args = parser.parse_args()

# parameters and utils
device = 'cuda'
ROOT = '/home/fcdl/dataset/'


# SNNLoss definition
def dist2(x):
    return torch.norm(x[:, None] - x, dim=2, p=2)


def dist(x):
    xn = x.norm(p=2, dim=1).pow(2)
    xn_t = xn.unsqueeze(0).t()
    xxT = torch.mm(x, x.permute(1, 0))
    dist = xn + xn_t - 2 * xxT
    return dist


# SNNLoss definition
class SNNLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, y, T=1):  # x 2-D matrix of BxF, y 1-D vector of B
        b = len(y)

        # print(x.norm(p=2))

        dist = dist2(x)

        # print(dist)
        # print(torch.pdist(x))

        # make diagonal mask
        m_den = 1 - torch.eye(b)
        # m_den = torch.log(m_den.float())
        m_den = m_den.float().to(x.device)

        e_dist = (-dist) * T

        den_dist = torch.clone(e_dist)
        den_dist[m_den == 0] = float('-inf')

        # make per class mask
        m_num = (y == y.unsqueeze(0).t()).type(torch.int) - torch.eye(b, dtype=torch.int).to(y.device)
        num_dist = torch.clone(e_dist)
        num_dist[m_num == 0] = float('-inf')

        # compute logsumexp
        num = torch.logsumexp(num_dist, dim=1)
        den = torch.logsumexp(den_dist, dim=1)

        # print(num, den)
        if torch.sum(torch.isinf(num)) > 0:
            num = num.clone()
            den = den.clone()
            num[torch.isinf(num)] = 0
            den[torch.isinf(num)] = 0

        if torch.sum(torch.isnan(num)) > 0:
            print(x.shape)
            print(x)
            print(num_dist.shape)
            print(num_dist)
            print(den_dist)
            print(num.shape)
            print(num)
            print(den)
            raise Exception()

        return -(num - den).mean()


def print_tsne(X,y):
    X_p = X
    plt.scatter(X_p[:, 0], X_p[:, 1], c=y, cmap=plt.get_cmap('tab10'))
    plt.colorbar()


def train_epoch_single(network, train_loader, optimizer):
    src_criterion = nn.CrossEntropyLoss()

    network.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    batch_idx = 0

    for batch in train_loader:

        optimizer.zero_grad()

        inputs, targets = batch

        inputs = inputs.to(device)
        targets = targets.to(device)

        logits, feat = network.forward(inputs)  # feature vector only
        prediction = network.predict(logits)  # class scores

        loss_bx = src_criterion(prediction, targets)  # CE loss

        loss_bx.backward()
        optimizer.step()

        # get predictions
        _, predicted = prediction.max(1)
        tr_tot = targets.size(0)
        tr_crc = predicted.eq(targets).sum().item()

        # compute statistics
        train_loss += loss_bx.item()
        train_total += tr_tot
        train_correct += tr_crc

        batch_idx += 1
        if batch_idx % 200 == 0:
            print(f"{batch_idx:3d} | Source Loss: {loss_bx:.6f} "
                  f"Source Acc : {100.0 * train_correct / train_total:.2f}")

    train_acc = 100. * train_correct / train_total

    return train_loss / batch_idx, train_acc


def train_epoch_dann(network, train_loader, optimizer, ALPHA=1, use_target_labels=True):
    src_criterion = nn.CrossEntropyLoss()
    dom_criterion = nn.BCEWithLogitsLoss()

    network.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    train_total_src = 0
    train_correct_src = 0
    batch_idx = 0
    # scheduler.step()

    for source_batch, target_batch in train_loader:

        p = float(batch_idx + start_steps) / total_steps
        lam = 2. / (1. + np.exp(-10 * p)) - 1

        optimizer.zero_grad()

        inputs, targets = source_batch

        inputs = inputs.to(device)
        targets = targets.to(device)  # ground truth class scores
        domains = torch.zeros(inputs.shape[0], 1).to(device)  # source is index 0

        logits, feat = network.forward(inputs)  # feature vector only
        prediction = network.predict(logits)  # class scores
        s_prediction = network.discriminate_domain(feat, lam)  # domain score

        loss_bx_src = src_criterion(prediction, targets)  # CE loss
        loss_bx_dom_s = dom_criterion(s_prediction, domains)

        _, predicted = prediction.max(1)
        tr_tot = targets.size(0)  # only on target
        tr_crc = predicted.eq(targets).sum().item()  # only on target

        train_total_src += tr_tot
        train_correct_src += tr_crc

        # train the target
        inputs, targets = target_batch

        inputs, targets = inputs.to(device), targets.to(device)  # class gt
        domains = torch.ones(inputs.shape[0], 1).to(device)  # target is index 1

        logits, feat = network.forward(inputs)  # feature vector only
        prediction = network.predict(logits)  # class scores
        d_prediction = network.discriminate_domain(feat, lam)  # domain score

        if use_target_labels:
            loss_bx_tar = src_criterion(prediction, targets)
        else:
            loss_bx_tar = 0.
        loss_bx_dom_t = dom_criterion(d_prediction, domains)

        # sum the losses and do backward propagation
        loss_dom = (loss_bx_dom_s + loss_bx_dom_t)
        loss_bx = loss_bx_src + loss_bx_tar + ALPHA * loss_dom  # use target labels

        loss_bx.backward()
        optimizer.step()

        _, predicted = prediction.max(1)
        tr_tot = targets.size(0)  # only on target
        tr_crc = predicted.eq(targets).sum().item()  # only on target

        # compute statistics
        train_loss += loss_bx.item()
        train_total += tr_tot
        train_correct += tr_crc

        batch_idx += 1
        if batch_idx % 200 == 0:
            print(f"Batch {batch_idx} / {len(train_loader)}\n\t"
                  f"Lambda {lam:.4f} "
                  f"Domain Loss: {loss_dom:.6f}\n\t"
                  f"Source Loss: {loss_bx_src:.6f} "
                  f"Source Acc : {100.0 * train_correct_src / train_total_src:.2f} "
                  f"SrcDom Acc : {1 - torch.sigmoid(s_prediction.detach()).mean().cpu().item():.3f}\n\t"
                  f"Target Loss: {loss_bx_tar:.6f} "
                  f"Target Acc : {100.0 * train_correct / train_total:.2f} "
                  f"TarDom Acc : {torch.sigmoid(d_prediction.detach()).cpu().mean().item():.3f}"
                  )

    train_acc = 100. * train_correct / train_total

    return train_loss / batch_idx, train_acc


def train_epoch_snnl(network, train_loader, optimizer, t_optim, T_d, T_c, ALPHA_Y=1., ALPHA_D=-1.):
    src_criterion = nn.CrossEntropyLoss()
    snnl = SNNLoss()

    network.train()
    train_loss = 0
    class_snnl_loss_cum = 0
    domain_snnl_loss_cum = 0
    train_correct = 0
    train_total = 0
    train_total_src = 0
    train_correct_src = 0
    batch_idx = 0
    # scheduler.step()

    for source_batch, target_batch in train_loader:

        optimizer.zero_grad()

        inputs_s, targets_s = source_batch

        inputs_s = inputs_s.to(device)
        targets_s = targets_s.to(device)  # ground truth class scores
        domain_s = torch.zeros(inputs_s.shape[0]).to(device)  # source is index 0

        logit_s, feat_s = network.forward(inputs_s)  # feature vector only
        prediction = network.predict(logit_s)  # class scores

        loss_bx_src = src_criterion(prediction, targets_s)  # CE loss

        _, predicted = prediction.max(1)
        tr_tot = targets_s.size(0)  # only on target
        tr_crc = predicted.eq(targets_s).sum().item()  # only on target

        train_total_src += tr_tot
        train_correct_src += tr_crc

        # train the target
        inputs_t, targets_t = target_batch

        inputs_t, targets_t = inputs_t.to(device), targets_t.to(device)  # class gt
        domain_t = torch.ones(inputs_t.shape[0]).to(device)  # target is index 1

        logit_t, feat_t = network.forward(inputs_t)  # feature vector only
        prediction = network.predict(logit_t)  # class scores

        loss_bx_tar = src_criterion(prediction, targets_t)

        _, predicted = prediction.max(1)
        tr_tot = targets_t.size(0)  # only on target
        tr_crc = predicted.eq(targets_t).sum().item()  # only on target

        # sum the CE losses
        loss_cl = loss_bx_src + loss_bx_tar

        logits = torch.cat((logit_s, logit_t), 0)
        feats = torch.cat((feat_s, feat_t), 0)
        targets = torch.cat((targets_s, targets_t), 0)
        domains = torch.cat((domain_s, domain_t), 0)

        class_snnl_loss = snnl(feats.reshape(feats.shape[0], -1), targets, T_c)
        domain_snnl_loss = snnl(feats.reshape(feats.shape[0], -1), domains, T_d)

        loss = loss_cl + ALPHA_D * domain_snnl_loss + ALPHA_Y * class_snnl_loss

        loss.backward()
        optimizer.step()

        t_optim.zero_grad()
        class_snnl_loss = snnl(feats.reshape(feats.shape[0], -1), targets, T_c)
        class_snnl_loss.backward()
        t_optim.step()

        t_optim.zero_grad()
        domain_snnl_loss = snnl(feats.detach().reshape(feats.shape[0], -1), domains, T_d)
        domain_snnl_loss.backward()
        t_optim.step()

        # compute statistics
        train_loss += loss_cl.item()
        class_snnl_loss_cum += class_snnl_loss.item()
        domain_snnl_loss_cum += domain_snnl_loss.item()
        train_total += tr_tot
        train_correct += tr_crc

        batch_idx += 1
        if batch_idx % 200 == 0 or batch_idx == 1:
            print(f"Batch {batch_idx} / {len(train_loader)}\n\t"
                  f"Source Loss: {loss_bx_src:.6f} "
                  f"Source Acc : {100.0 * train_correct_src / train_total:.2f} "
                  f"Target Loss: {loss_bx_tar:.6f} "
                  f"Target Acc : {100.0 * train_correct / train_total:.2f}\n\t "
                  f"Class loss: {class_snnl_loss_cum / batch_idx:.6f} "
                  f"Domain loss: {domain_snnl_loss_cum / batch_idx:.6f} "
                  f"Td: {T_d.item():.6f}")

    train_acc = 100. * train_correct / train_total

    return train_loss / batch_idx, train_acc


def minimize_T(network, train_loader, optimizer, T):
    snnl = SNNLoss()

    for source_batch, target_batch in train_loader:
        optimizer.zero_grad()

        # compute sources
        inputs_s, targets_s = source_batch
        inputs_s, targets_s = inputs_s.to(device), targets_s.to(device)
        domain_s = torch.zeros(inputs_s.shape[0]).to(device)  # source is index 0

        logit_s, feat_s = network.forward(inputs_s)  # feature vector only

        # computes targets
        inputs_t, targets_t = target_batch

        inputs_t, targets_t = inputs_t.to(device), targets_t.to(device)  # class gt
        domain_t = torch.ones(inputs_t.shape[0]).to(device)  # target is index 1
        logit_t, feat_t = network.forward(inputs_t)  # feature vector only

        logits = torch.cat((logit_s, logit_t), 0).detach()
        feats = torch.cat((feat_s, feat_t), 0).detach()
        targets = torch.cat((targets_s, targets_t), 0).detach()
        domains = torch.cat((domain_s, domain_t), 0).detach()

        # class_snnl_loss = snnl(feats.reshape(feats.shape[0], -1), targets, T_c)
        domain_snnl_loss = snnl(feats.reshape(feats.shape[0], -1), domains, T)
        domain_snnl_loss.backward()

        optimizer.step()

    print(T)


def valid(network, valid_loader):
    criterion = nn.CrossEntropyLoss()
    snnl = SNNLoss()
    # make validation
    network.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    domain_acc = 0
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs, feats = network.forward(inputs)
            predictions = network.predict(outputs)  # class score
            domains = network.discriminate_domain(feats, 0)  # domain score (correct if 1., 0.5 is wanted)

            loss_bx = criterion(predictions, targets)

            test_loss += loss_bx.item()
            _, predicted = predictions.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()

            domain_acc += torch.sigmoid(domains.cpu().detach()).sum().item()

    # normalize and print stats
    test_acc = 100. * test_correct / test_total
    domain_acc = 100. * domain_acc / test_total
    test_loss /= len(valid_loader)

    return test_loss, test_acc, domain_acc


if __name__=='__main__':
    # Make the dataset
    transform = tv.transforms.Compose([transforms.Resize((28, 28)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    augmentation = tv.transforms.Compose([transforms.RandomCrop(28)])

    source = tv.datasets.MNIST(ROOT, train=True, download=True,
                               transform=tv.transforms.Compose([
                                   tv.transforms.Grayscale(3),
                                   transform])
                               )

    test = MNISTM(ROOT, train=False, download=True, transform=transform)

    target = MNISTM(ROOT, train=True, download=True, transform=transform)

    target_loader = DataLoader(target, batch_size=128, shuffle=True, num_workers=8)
    test_loader = DataLoader(test, batch_size=128, shuffle=False, num_workers=8)
    source_loader = DataLoader(source, batch_size=128, shuffle=True, num_workers=8)

    # make the hybrid dataset here.
    unsda = DoubleDataset(source, target)
    unsda_loader = DataLoader(unsda, 64, True, num_workers=8)

    # get mnist [0:4], mnistm[4:9]
    indices = get_index_of_classes(target.targets, list(range(0, 5)))
    half_target = Subset(target, indices)

    indices = get_index_of_classes(source.targets, list(range(5, 10)))
    half_source = Subset(source, indices)

    mixed = DoubleDataset(half_source, half_target)
    mixed_loader = DataLoader(mixed, 64, True, num_workers=8, drop_last=True)

    # define network
    net = lenet_net().to(device)

    # CE train loop!
    train_loader = mixed_loader

    EPOCHS = 40
    total_steps = EPOCHS * len(train_loader)

    print("Do a validation before starting to check it is ok...")
    val_loss, val_acc, dom_acc = valid(net, valid_loader=test_loader)
    print(f"Epoch {-1:03d} : Test Loss {val_loss:.6f}, Test Acc {val_acc:.2f}, Domain Acc {dom_acc:.2f}")
    print("Result should be random guessing, i.e. 10% accuracy")

    T_d = nn.Parameter(torch.FloatTensor([1]).to(device))
    T_c = nn.Parameter(torch.FloatTensor([1]).to(device))

    # training loop
    for epoch in range(EPOCHS):
        # steps
        start_steps = epoch * len(train_loader)

        # train epoch
        learning_rate = 0.01 / ((1 + 10 * (epoch) / EPOCHS) ** 0.75)
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
        t_optim = optim.SGD([T_d, T_c], lr=learning_rate)
        # scheduler.step()
        print(f"Learning rate: {learning_rate}")

        # minimize_T(net, train_loader, t_optim, T_d)

        train_loss, train_acc = train_epoch_snnl(net, train_loader=train_loader, optimizer=optimizer,
                                                 t_optim=t_optim, T_d=T_d, T_c=T_c, ALPHA_Y=args.Y, ALPHA_D=args.D)
        #train_loss, train_acc = train_epoch_dann(net, train_loader=train_loader, optimizer=optimizer)

        # valid!
        val_loss, val_acc, dom_acc = valid(net, valid_loader=test_loader)
        print(f"Epoch {epoch + 1:03d} : Test Loss {val_loss:.6f}, Test Acc {val_acc:.2f}, Domain Acc {dom_acc:.2f}")
        if train_loss < 1e-4:
            break

    torch.save(net.state_dict(), args.name + ".pth")
