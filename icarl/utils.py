import numpy as np
from scipy.spatial.distance import cdist
import torch
import torchvision
import os
import sys
import time
import math

device = 'cuda'


############################## Batch iterator ###############################

def iterate_minibatches(inputs, targets, batchsize, pixel_means, shuffle=False, augment=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if augment:
            # as in paper : 
            # pad feature arrays with 4 pixels on each side
            # and do random cropping of 32x32
            padded = np.pad(inputs[excerpt], ((0, 0), (0, 0), (4, 4), (4, 4)), mode='constant')
            random_cropped = np.zeros(inputs[excerpt].shape, dtype=np.float32)
            crops = np.random.random_integers(0, high=8, size=(batchsize, 2))
            for r in range(batchsize):
                # Cropping and possible flipping
                if (np.random.randint(2) > 0):
                    random_cropped[r, :, :, :] = padded[r, :, crops[r, 0]:(crops[r, 0] + 32),
                                                 crops[r, 1]:(crops[r, 1] + 32)]
                else:
                    random_cropped[r, :, :, :] = padded[r, :, crops[r, 0]:(crops[r, 0] + 32),
                                                 crops[r, 1]:(crops[r, 1] + 32)][:, :, ::-1]
            inp_exc = random_cropped
        else:
            inp_exc = inputs[excerpt]

        yield np.array(inp_exc, dtype=np.float32), targets[excerpt]


def accuracy_measure(X_valid, Y_valid, pixel_means, class_means, network, top1_acc_list, iteration, iteration_total,
                     type_data):
    stat_hb1 = []
    stat_icarl = []
    stat_ncm = []

    for batch in iterate_minibatches(X_valid, Y_valid, min(500, len(X_valid)), pixel_means, shuffle=False):
        inputs, targets_prep = batch
        targets = np.zeros((inputs.shape[0], 100), np.float32)
        targets[range(len(targets_prep)), targets_prep.astype('int32')] = 1.
        inputs = torch.tensor(inputs).to(device)
        outputs = network.forward(inputs)
        pred = network.predict(outputs).cpu().detach().numpy()
        outputs = outputs.cpu().detach().numpy()

        outputs = (outputs.T / np.linalg.norm(outputs.T, axis=0)).T

        # Compute score for iCaRL
        sqd = cdist(class_means[:, :, 0].T, outputs, 'sqeuclidean')
        score_icarl = (-sqd).T
        # Compute score for NCM
        sqd = cdist(class_means[:, :, 1].T, outputs, 'sqeuclidean')
        score_ncm = (-sqd).T

        # Compute the accuracy over the batch
        stat_hb1 += ([ll in best for ll, best in zip(targets_prep, np.argsort(pred, axis=1)[:, -1:])])
        stat_icarl += ([ll in best for ll, best in zip(targets_prep, np.argsort(score_icarl, axis=1)[:, -1:])])
        stat_ncm += ([ll in best for ll, best in zip(targets_prep, np.argsort(score_ncm, axis=1)[:, -1:])])

    print("Final results on " + type_data + " classes:")
    print("  top 1 accuracy iCaRL          :\t\t{:.2f} %".format(np.average(stat_icarl) * 100))
    print("  top 1 accuracy Hybrid 1       :\t\t{:.2f} %".format(np.average(stat_hb1) * 100))
    print("  top 1 accuracy NCM            :\t\t{:.2f} %".format(np.average(stat_ncm) * 100))

    top1_acc_list[iteration, 0, iteration_total] = np.average(stat_icarl) * 100
    top1_acc_list[iteration, 1, iteration_total] = np.average(stat_hb1) * 100
    top1_acc_list[iteration, 2, iteration_total] = np.average(stat_ncm) * 100

    return top1_acc_list


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(40 - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(40 - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
