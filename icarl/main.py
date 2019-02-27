# THEANO_FLAGS='cuda.root=/usr/local/cuda,device=gpu1,floatX=float32,lib.cnmem=0.09' python
from __future__ import print_function
import sys
sys.path.append('.')
import time
import numpy as np
from icarl import utils

import torch
import torchvision
import torch.nn as nn
import copy
from networks import networks
import torch.optim as optim
from icarl.utils import progress_bar

import torch.nn.functional as F

######### Modifiable Settings ##########
batch_size = 128  # Batch size
n = 5  # Set the depth of the architecture: n = 5 -> 32 layers (See He et al. paper)
nb_cl = 10  # Classes per group
nb_protos = 20  # Number of prototypes per class at the end: total protoset memory/ total number of classes
epochs = 70  # Total number of epochs
lr_old = 2.  # Initial learning rate
lr_strat = [49, 63]  # Epochs where learning rate gets decreased
lr_factor = 5.  # Learning rate decrease factor
wght_decay = 0.00001  # Weight Decay
nb_runs = 10  # Number of runs (random ordering of classes at each run)
np.random.seed(1993)  # Fix the random seed

########################################

device = 'cuda'


def save_checkpoint(state, filename):
    torch.save(state, filename)


# Load the dataset
XY_train_total = torchvision.datasets.CIFAR100(root='../data', train=True, download=True)
Y_train_total = XY_train_total.train_labels
X_train_total = XY_train_total.train_data.transpose(0, 3, 1, 2) / np.float32(255)  # what?

pixel_means = np.mean(X_train_total, axis=0).reshape(1, 32, 32, 3).transpose(0, 3, 1, 2)

XY_valid_total = torchvision.datasets.CIFAR100(root='../data', train=False, download=True)
Y_valid_total = XY_valid_total.test_labels
X_valid_total = XY_valid_total.test_data.transpose(0, 3, 1, 2) / np.float32(255)

X_train_total -= pixel_means
X_valid_total -= pixel_means

# Initialization of accuracy lists (3 methods, 10 groups, n_runs) + dictionary=number of training samples per class
dictionary_size = 500
top1_acc_list_cumul = np.zeros((100 / nb_cl, 3, nb_runs))
top1_acc_list_ori = np.zeros((100 / nb_cl, 3, nb_runs))

full_order = np.load('fixed_order.npy')
# Launch the different runs 
for iteration_total in range(nb_runs):

    # Select the order for the class learning 
    order = full_order[iteration_total]

    # Prepare Theano variables for inputs and targets
    # Create neural network model
    print('Run {0} starting ...'.format(iteration_total))

    network = networks.CifarResNet()
    network = network.to(device)
    loss = nn.BCEWithLogitsLoss(size_average=True)

    lr = lr_old

    # Initialization of the variables for this run
    X_valid_cumuls = []
    X_protoset_cumuls = []
    X_train_cumuls = []
    Y_valid_cumuls = []
    Y_protoset_cumuls = []
    Y_train_cumuls = []
    acc_cumuls = [[], [], []]
    acc_original = [[], [], []]
    alpha_dr_herding = np.zeros((100 / nb_cl, dictionary_size, nb_cl), np.float32)  # ?

    # The following contains all the training samples of the different classes 
    # because we want to compare our method with the theoretical case where all the training samples are stored
    prototypes = np.zeros(
        (100, dictionary_size, X_train_total[0].shape[0], X_train_total[0].shape[1], X_train_total[0].shape[2]),
        dtype=np.float32)
    for orde in range(100):
        prototypes[orde, :, :, :, :] = X_train_total[np.where(Y_train_total == order[orde])]

    for iteration in range(100 // nb_cl):
        # Save data results at each increment
        np.save(sys.argv[1] + 'top1_acc_list_cumul_icarl_cl' + str(nb_cl), top1_acc_list_cumul)
        np.save(sys.argv[1] + 'top1_acc_list_ori_icarl_cl' + str(nb_cl), top1_acc_list_ori)

        # Prepare the training data for the current batch of classes
        actual_cl = order[range(iteration * nb_cl, (iteration + 1) * nb_cl)]
        indices_train_10 = np.array(
            # select index i if example i is in actual classes
            [i in order[range(iteration * nb_cl, (iteration + 1) * nb_cl)] for i in Y_train_total]).reshape(-1)
        indices_test_10 = np.array(
            [i in order[range(iteration * nb_cl, (iteration + 1) * nb_cl)] for i in Y_valid_total]).reshape(-1)

        X_train = X_train_total[indices_train_10]
        X_valid = X_valid_total[indices_test_10]
        X_valid_cumuls.append(X_valid)
        X_train_cumuls.append(X_train)
        X_valid_cumul = np.concatenate(X_valid_cumuls)
        X_train_cumul = np.concatenate(X_train_cumuls)
        Y_train = np.array(Y_train_total)[indices_train_10]
        Y_valid = np.array(Y_valid_total)[indices_test_10]
        Y_valid_cumuls.append(Y_valid)
        Y_train_cumuls.append(Y_train)
        Y_valid_cumul = np.concatenate(Y_valid_cumuls)
        Y_train_cumul = np.concatenate(Y_train_cumuls)

        # Add the stored exemplars to the training data
        if iteration == 0:
            X_valid_ori = X_valid
            Y_valid_ori = Y_valid
        else:
            # make the combined training set from exemplar and current samples
            X_protoset = np.concatenate(X_protoset_cumuls)
            Y_protoset = np.concatenate(Y_protoset_cumuls)
            X_train = np.concatenate((X_train, X_protoset), axis=0)
            Y_train = np.concatenate((Y_train, Y_protoset))

        # Launch the training loop
        new_lr = lr_old
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, network.parameters()), lr=new_lr, momentum=0.9,
                              weight_decay=wght_decay, nesterov=False)

        print("\n")
        print('Batch of classes number {0} arrives ...'.format(iteration + 1))
        for epoch in range(epochs):

            print('\nEpoch: %d' % epoch)
            network.train()
            train_loss = 0
            correct = 0
            total = 0

            # Shuffle training data
            train_indices = np.arange(len(X_train))
            np.random.shuffle(train_indices)
            X_train = X_train[train_indices, :, :, :]
            Y_train = Y_train[train_indices]
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            count = 0
            for batch in utils.iterate_minibatches(X_train, Y_train, batch_size, pixel_means, shuffle=True,
                                                   augment=True):
                # that's like a custom Data Loader
                count += 1
                inputs, targets_prep = batch

                targets = np.zeros((inputs.shape[0], 100), np.float32)
                targets[range(len(targets_prep)), targets_prep.astype('int32')] = 1.
                inputs = torch.tensor(inputs).to(device)

                optimizer.zero_grad()
                outputs = network.forward(inputs)  # feature vector only
                prediction = network.predict(outputs)  # make the prediction with sigmoid, NCM or other
                targets = torch.tensor(targets).to(outputs.device)
                targets_prep = torch.LongTensor(targets_prep).to(outputs.device)

                if iteration > 0:  # apply distillation
                    outputs_old = network2.forward(inputs)
                    prediction_old = network2.predict(outputs_old)
                    targets[:, np.array(order[range(0, iteration * nb_cl)])] = \
                        F.sigmoid(prediction_old[:, np.array(order[range(0, iteration * nb_cl)])])

                loss_bx = loss(prediction, targets)  # joins classification and distillation losses
                loss_bx.backward()
                optimizer.step()

                train_loss += loss_bx.item()
                _, predicted = prediction.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets_prep).sum().item()
                if count % 20 == 0:
                    print('\n')
                    progress_bar(count, len(XY_train_total), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                 % (train_loss / (count), 100. * correct / total, correct, total))

            train_batches = len(XY_train_total) # non mi e' chiarissimo il senso di questa variabile

            # And a full pass over the validation data:

            network.eval()
            test_loss = 0
            correct = 0
            total = 0
            count = 0
            for batch in utils.iterate_minibatches(X_valid, Y_valid, min(500, len(X_valid)), pixel_means,
                                                   shuffle=False):
                count += 1
                inputs, targets_prep = batch
                targets = np.zeros((inputs.shape[0], 100), np.float32)
                targets[range(len(targets_prep)), targets_prep.astype('int32')] = 1.
                inputs = torch.tensor(inputs).to(device)
                outputs = network.forward(inputs)
                outputs = network.predict(outputs)
                targets = torch.tensor(targets).to(outputs.device)
                loss_bx = loss(outputs, targets)
                targets_prep = torch.LongTensor(targets_prep).to(outputs.device)
                test_loss += loss_bx.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets_prep).sum().item()
                if count % 40 == 0:
                    progress_bar(count, len(XY_valid_total), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                 % (test_loss / (count), 100. * correct / total, correct, total))

            acc = 100. * correct / total

            # adjust learning rate
            if (epoch + 1) in lr_strat:
                new_lr = new_lr / lr_factor
                print("New LR:" + str(new_lr))
                optimizer = optim.SGD(filter(lambda p: p.requires_grad, network.parameters()), lr=new_lr, momentum=0.9,
                                      weight_decay=wght_decay, nesterov=False)

        # epochs ended for this class-batch
        # Duplicate current network to distillate info
        network2 = copy.deepcopy(network)
        network2.eval()

        # Save the network

        save_checkpoint({
            'iteration': iteration,
            'state_dict': network.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, "iter_" + str(iteration) + "_checkpoint.pth.tar")

        # Update Exemplars
        nb_protos_cl = int(np.ceil(nb_protos * 100. / nb_cl / (iteration + 1)))  # num of exemplars per class
        # Herding
        print('Updating exemplar set...')
        network.eval()
        for iter_dico in range(nb_cl):
            # Possible exemplars in the feature space and projected on the L2 sphere
            # exemplars of class iter_dico + nb_cl
            pinput = torch.tensor(np.array((prototypes[iteration * nb_cl + iter_dico]), dtype=np.float32)).to(device)
            mapped_prototypes = network.forward(pinput).cpu().detach().numpy()
            D = mapped_prototypes.T
            D = D / np.linalg.norm(D, axis=0)
            # Herding procedure : ranking of the potential exemplars
            mu = np.mean(D, axis=1)
            # set exemplar to zero
            alpha_dr_herding[iteration, :, iter_dico] = alpha_dr_herding[iteration, :, iter_dico] * 0
            w_t = mu
            iter_herding = 0
            iter_herding_eff = 0
            # Herding algorithm
            while not (np.sum(alpha_dr_herding[iteration, :, iter_dico] != 0) == min(nb_protos_cl,
                                                                                     500)) and iter_herding_eff < 1000:
                tmp_t = np.dot(w_t, D)
                ind_max = np.argmax(tmp_t)
                iter_herding_eff += 1
                if alpha_dr_herding[iteration, ind_max, iter_dico] == 0:
                    alpha_dr_herding[iteration, ind_max, iter_dico] = 1 + iter_herding
                    iter_herding += 1
                w_t = w_t + mu - D[:, ind_max]

        # Prepare the protoset
        X_protoset_cumuls = []
        Y_protoset_cumuls = []

        # Class means for iCaRL and NCM + Storing the selected exemplars in the protoset
        print('Computing mean-of_exemplars and theoretical mean...')
        class_means = np.zeros((64, 100, 2))

        for iteration2 in range(iteration + 1):
            current_cl = order[range(iteration2 * nb_cl, (iteration2 + 1) * nb_cl)]
            for iter_dico in range(nb_cl):

                pinput = torch.tensor(np.array((prototypes[iteration2 * nb_cl + iter_dico]), dtype=np.float32)).to(
                    device)

                # Collect data in the feature space for each class
                mapped_prototypes = network.forward(pinput).cpu().detach().numpy()
                D = mapped_prototypes.T
                D = D / np.linalg.norm(D, axis=0)

                # Flipped version also
                inverted = np.array(prototypes[iteration2 * nb_cl + iter_dico][:, :, :, ::-1])
                pinput2 = torch.tensor(np.array((inverted - pixel_means), dtype=np.float32)).to(device)
                mapped_prototypes2 = network.forward(pinput2).cpu().detach().numpy()
                D2 = mapped_prototypes2.T
                D2 = D2 / np.linalg.norm(D2, axis=0)

                # iCaRL
                alph = alpha_dr_herding[iteration2, :, iter_dico]
                alph = (alph > 0) * (alph < nb_protos_cl + 1) * 1.
                X_protoset_cumuls.append(prototypes[iteration2 * nb_cl + iter_dico, np.where(alph == 1)[0]])
                Y_protoset_cumuls.append(order[iteration2 * nb_cl + iter_dico] * np.ones(len(np.where(alph == 1)[0])))
                alph = alph / np.sum(alph)
                class_means[:, current_cl[iter_dico], 0] = (np.dot(D, alph) + np.dot(D2, alph)) / 2
                class_means[:, current_cl[iter_dico], 0] /= np.linalg.norm(class_means[:, current_cl[iter_dico], 0])

                # Normal NCM
                alph = np.ones(dictionary_size) / dictionary_size
                class_means[:, current_cl[iter_dico], 1] = (np.dot(D, alph) + np.dot(D2, alph)) / 2
                class_means[:, current_cl[iter_dico], 1] /= np.linalg.norm(class_means[:, current_cl[iter_dico], 1])

        np.save('cl_means', class_means)

        # Calculate validation error of model on the first nb_cl classes:
        print('Computing accuracy on the original batch of classes...')
        top1_acc_list_ori = utils.accuracy_measure(X_valid_ori, Y_valid_ori, pixel_means, class_means, network,
                                                   top1_acc_list_ori, iteration, iteration_total, 'original')
        names = ['iCaRL', 'Hybrid', 'NCM']

        # Calculate validation error of model on the cumul of classes:
        print('Computing cumulative accuracy...')
        top1_acc_list_cumul = utils.accuracy_measure(X_valid_cumul, Y_valid_cumul, pixel_means, class_means, network,
                                                     top1_acc_list_cumul, iteration, iteration_total, 'cumul of')

# Final save of the data
np.save(sys.argv[1] + 'top1_acc_list_cumul_icarl_cl' + str(nb_cl), top1_acc_list_cumul)
np.save(sys.argv[1] + 'top1_acc_list_ori_icarl_cl' + str(nb_cl), top1_acc_list_ori)
