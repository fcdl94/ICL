import torch
import torch.nn as nn
import numpy as np
from snnl import SNNLoss

device = 'cuda'


def train_epoch_dann(network, start_steps, total_steps, train_loader, optimizer, ALPHA=1, use_target_labels=True):
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
        s_prediction, _ = network.discriminate_domain(feat, lam)  # domain score

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
        d_prediction, _ = network.discriminate_domain(feat, lam)  # domain score

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

        if loss_bx.item() >= 10:
            print(batch_idx, loss_bx_src, loss_bx_tar, loss_dom)

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


def train_epoch_snnl(network, start_steps, total_steps, train_loader, optimizer, t_o, T_d, T_c, ALPHA_Y=0, ALPHA_D=-1, use_target_labels=False):
    src_criterion = nn.CrossEntropyLoss()
    snnl_inv = SNNLoss(inv=True)
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

    for source_batch, target_batch in train_loader:

        p = float(batch_idx + start_steps) / total_steps
        lam = 2. / (1. + np.exp(-5 * p)) - 1

        optimizer.zero_grad()

        inputs_s, targets_s = source_batch

        inputs_s = inputs_s.to(device)
        targets_s = targets_s.to(device)  # ground truth class scores
        domain_s = torch.zeros(inputs_s.shape[0]).to(device)  # source is index 0

        logit_s, feat_s = network.forward(inputs_s)  # feature vector only
        prediction = network.predict(logit_s)  # class scores
        # d_prediction_s = network.discriminate_domain(feat_s)  # domain score

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
        # d_prediction_t = network.discriminate_domain(feat_t)  # domain score

        if use_target_labels:
            loss_bx_tar = src_criterion(prediction, targets_t)
        else:
            loss_bx_tar = 0.

        _, predicted = prediction.max(1)
        tr_tot = targets_t.size(0)  # only on target
        tr_crc = predicted.eq(targets_t).sum().item()  # only on target

        # sum the CE losses
        loss_cl = (loss_bx_src + loss_bx_tar)

        logits = torch.cat((logit_s, logit_t), 0)
        feats = torch.cat((feat_s, feat_t), 0)
        # d_prediction = torch.cat((d_prediction_s, d_prediction_t), 0)
        targets = torch.cat((targets_s, targets_t), 0)
        domains = torch.cat((domain_s, domain_t), 0)

        class_snnl_loss = snnl(logits, targets, T_c)
        domain_snnl_loss = snnl_inv(feats, domains, T_d)

        loss = loss_cl + lam * ALPHA_D * domain_snnl_loss + ALPHA_Y * class_snnl_loss

        loss.backward()
        optimizer.step()

        # t_optim.zero_grad()
        # class_snnl_losst = snnl(feats.detach(), targets, T_c)
        # class_snnl_losst.backward()
        # t_optim.step()
        # t_optim.zero_grad()
        # domain_snnl_losst = snnl(feats.detach(), domains, T_d)
        # domain_snnl_losst.backward()
        # t_optim.step()

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
                  f"Source Acc : {100.0 * train_correct_src / train_total_src:.2f} "
                  f"Target Loss: {loss_bx_tar:.6f} "
                  f"Target Acc : {100.0 * train_correct / train_total:.2f}\n\t "
                  f"Class loss: {(class_snnl_loss_cum / batch_idx):.6f} "
                  f"Domain loss: {(domain_snnl_loss_cum / batch_idx):.6f} "
                  f"Td: {T_d.item():.3f} "
                  f"Alpha: {lam * ALPHA_D:.3f}")

    train_acc = 100. * train_correct / train_total

    return train_loss / batch_idx, train_acc


def valid(network, valid_loader, conf_matrix=False):
    criterion = nn.CrossEntropyLoss()
    # make validation
    network.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    domain_acc = 0

    targets_cum = []
    predict_cum = []

    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs, feats = network.forward(inputs)
            predictions = network.predict(outputs)  # class score
            domains, _ = network.discriminate_domain(feats, 0)  # domain score (correct if 1., 0.5 is wanted)

            loss_bx = criterion(predictions, targets)

            test_loss += loss_bx.item()
            _, predicted = predictions.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()

            targets_cum.append(targets)
            predict_cum.append(predicted)

            domain_acc += torch.sigmoid(domains.cpu().detach()).sum().item()

    # normalize and print stats
    test_acc = 100. * test_correct / test_total
    domain_acc = 100. * domain_acc / test_total
    test_loss /= len(valid_loader)

    return test_loss, test_acc, domain_acc


def train_epoch_dann_dg(network, start_steps, total_steps, train_loader, optimizer, ALPHA=1, use_target_labels=True):
    src_criterion = nn.CrossEntropyLoss()
    dom_criterion = nn.CrossEntropyLoss()

    network.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    batch_idx = 0

    for batch in train_loader:

        p = float(batch_idx + start_steps) / total_steps
        lam = 2. / (1. + np.exp(-10 * p)) - 1

        optimizer.zero_grad()

        inputs, targets, domains = batch
        inputs, targets, domains = inputs.to(device), targets.to(device), domains.to(device)

        logits, feat = network.forward(inputs)  # feature vector only
        prediction = network.predict(logits)  # class scores
        s_prediction, _ = network.discriminate_domain(feat, lam)  # domain score

        loss_bx = src_criterion(prediction, targets)  # CE loss
        loss_dom = dom_criterion(s_prediction, domains)

        _, predicted = prediction.max(1)
        tr_tot = targets.size(0)  # only on target
        tr_crc = predicted.eq(targets).sum().item()  # only on target

        loss = loss_bx + ALPHA * loss_dom  # use target labels

        loss.backward()
        optimizer.step()

        # compute statistics
        train_loss += loss_bx.item()
        train_total += tr_tot
        train_correct += tr_crc

        if loss_bx.item() >= 10:
            print(batch_idx, loss_bx, loss_dom)

        batch_idx += 1
        if batch_idx % 200 == 0:
            print(f"Batch {batch_idx} / {len(train_loader)}\n\t"
                  f"Lambda {lam:.4f} "
                  f"Domain Loss: {loss_dom:.6f}\n\t"
                  f"Source Loss: {loss_bx:.6f} "
                  f"Source Acc : {100.0 * train_correct / train_total:.2f}")

    train_acc = 100. * train_correct / train_total

    return train_loss / batch_idx, train_acc


def train_epoch_snnl_dg(network, start_steps, total_steps, train_loader, optimizer,
                        t_o, T_d, T_c, ALPHA_Y=0, ALPHA_D=-1, use_target_labels=False):

    src_criterion = nn.CrossEntropyLoss()
    snnl_inv = SNNLoss(inv=True)
    snnl = SNNLoss()

    network.train()
    train_loss = 0
    class_snnl_loss_cum = 0
    domain_snnl_loss_cum = 0
    train_correct = 0
    train_total = 0
    batch_idx = 0

    for batch in train_loader:

        p = float(batch_idx + start_steps) / total_steps
        lam = 2. / (1. + np.exp(-10 * p)) - 1

        optimizer.zero_grad()

        inputs, targets, domains = batch
        inputs, targets, domains = inputs.to(device), targets.to(device), domains.to(device)

        logits, feats = network.forward(inputs)  # feature vector only
        prediction = network.predict(logits)  # class scores

        loss_cl = src_criterion(prediction, targets)  # CE loss

        _, predicted = prediction.max(1)
        tr_tot = targets.size(0)  # only on target
        tr_crc = predicted.eq(targets).sum().item()  # only on target

        class_snnl_loss = snnl(feats, targets, T_c)
        domain_snnl_loss = snnl_inv(feats, domains, T_d)

        loss = loss_cl + lam * ALPHA_D * domain_snnl_loss + ALPHA_Y * class_snnl_loss

        loss.backward()
        optimizer.step()

        # t_optim.zero_grad()
        # class_snnl_losst = snnl(feats.detach(), targets, T_c)
        # class_snnl_losst.backward()
        # t_optim.step()
        # t_optim.zero_grad()
        # domain_snnl_losst = snnl(feats.detach(), domains, T_d)
        # domain_snnl_losst.backward()
        # t_optim.step()

        # compute statistics
        train_loss += loss_cl.item()
        class_snnl_loss_cum += class_snnl_loss.item()
        domain_snnl_loss_cum += domain_snnl_loss.item()
        train_total += tr_tot
        train_correct += tr_crc

        batch_idx += 1
        if batch_idx % 200 == 0 or batch_idx == 1:
            print(f"Batch {batch_idx} / {len(train_loader)}\n\t"
                  f"Source Loss: {loss_cl:.6f} "
                  f"Source Acc : {100.0 * train_correct / train_total:.2f}\n\t"
                  f"Class loss: {(class_snnl_loss_cum / batch_idx):.6f} "
                  f"Domain loss: {(domain_snnl_loss_cum / batch_idx):.6f} "
                  f"Td: {T_d.item():.3f} "
                  f"Alpha: {lam * ALPHA_D:.3f}")

    train_acc = 100. * train_correct / train_total

    return train_loss / batch_idx, train_acc
