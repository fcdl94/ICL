from torch import tensor
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from scipy.spatial.distance import cdist

LR = 0.01
MEM_SIZE = 20*100
DECAY = 0.00001
EPOCHS = 70
LR_FACTOR = 5.


class ICarl:

    def __init__(self, network, n_classes=100, dictionary_size=100, mem_size=MEM_SIZE,
                 lr_init=LR, decay=DECAY, epochs=EPOCHS,
                 device='cuda'):
        """
        :param network: the backbone neural network of the model
        :param n_classes: Number of classes of the dataset
        :param dictionary_size: Number of example in each class (must be equal for all classes)
        :param mem_size: Number of prototypes to be stored
        :param lr_init: Initial learning rate (decayed by 5 after 0.7 and 0.9 epochs
        :param decay: weight decay for SGD
        :param epochs: number of epochs to train the model
        :param device: "cuda" or "cpu"
        """
        self.network = network.to(device)
        self.network2 = self.network

        self.loss = nn.BCEWithLogitsLoss(reduction='mean')

        self.device = device
        self.dataset = None
        self.nb_cl = None

        self.lr_init = lr_init
        self.mem_size = mem_size
        self.n_classes = n_classes
        self.dictionary_size = dictionary_size
        self.decay = decay
        self.epochs = epochs
        self.lr_strat = [epochs//7, epochs//9]
        self.lr_factor = LR_FACTOR

        self.prototypes = None

        self.alpha_dr_herding = np.zeros((n_classes, dictionary_size), np.float32)

    def fit(self, dataset, nb_cl):
        self.alpha_dr_herding = np.zeros((self.n_classes, self.dictionary_size), np.float32)

        self.dataset = dataset
        self.nb_cl = nb_cl

        x_protoset_cumuls = []
        y_protoset_cumuls = []

        for iteration in range(self.n_classes // nb_cl):
            # Add the stored exemplars to the training data
            if iteration > 0:
                x_protoset = np.concatenate(x_protoset_cumuls)
                y_protoset = np.concatenate(y_protoset_cumuls)
            else:
                x_protoset = None
                y_protoset = None

            # Prepare the training data for the current batch of classes
            dataset.next_iteration(x_protoset, y_protoset)

            # TRAIN THIS ITERATION #
            print('Batch of classes number {0} arrives ...'.format(iteration + 1))

            self.incremental_fit(iteration)  # train for N epochs (after each epoch validate)

            # Save the network
            torch.save({
                'iteration': iteration,
                'state_dict': self.network.state_dict()
            }, "iter_" + str(iteration) + "_checkpoint.pth.tar")
            # END OF TRAINING FOR THIS ITERATION #

            # UPDATE EXEMPLARS #
            print('Updating exemplar set...')
            x_protoset_cumuls, y_protoset_cumuls = self.update_exemplars(iteration)

            ## COMPUTE ACCURACY ##
            acc_cum = self.test(iteration + 1)

            print("Cumulative results")
            print("  top 1 accuracy iCaRL          :\t{:.2f} %".format(acc_cum[0]))
            print("  top 1 accuracy Hybrid 1       :\t{:.2f} %".format(acc_cum[1]))
            print("  top 1 accuracy NCM            :\t{:.2f} %".format(acc_cum[2]))

            acc_base = self.test(1)

            print("First batch results")
            print("  top 1 accuracy iCaRL          :\t{:.2f} %".format(acc_base[0]))
            print("  top 1 accuracy Hybrid 1       :\t{:.2f} %".format(acc_base[1]))
            print("  top 1 accuracy NCM            :\t{:.2f} %".format(acc_base[2]))

            print("")

    def incremental_fit(self, iteration):
        new_lr = self.lr_init
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.network.parameters()), lr=new_lr, momentum=0.9,
                              weight_decay=self.decay, nesterov=False)

        for epoch in range(self.epochs):
            self.network.train()
            train_loss = 0
            correct = 0
            total = 0

            # In each epoch, we do a full pass over the training data:
            for inputs, targets_prep in self.dataset.minibatches(augment=True):

                targets = np.zeros((inputs.shape[0], 100), np.float32)  # 100 = classes of cifar
                targets[range(len(targets_prep)), targets_prep.type(torch.int32)] = 1.  # prepare target for CE loss

                inputs = inputs.to(self.device)

                optimizer.zero_grad()
                outputs = self.network.forward(inputs)  # feature vector only
                prediction = self.network.predict(outputs)  # make the prediction with sigmoid, making g_y(xi)
                targets = tensor(targets).to(outputs.device)
                targets_prep = torch.LongTensor(targets_prep).to(outputs.device)

                if iteration > 0:  # apply distillation
                    outputs_old = self.network2.forward(inputs)
                    prediction_old = self.network2.predict(outputs_old)
                    targets[:, np.array(self.dataset.order[range(0, iteration * self.nb_cl)])] = \
                        torch.sigmoid(prediction_old[:, np.array(self.dataset.order[range(0, iteration * self.nb_cl)])])

                loss_bx = self.loss(prediction, targets)  # joins classification and distillation losses
                loss_bx.backward()
                optimizer.step()

                train_loss += loss_bx.item()
                _, predicted = prediction.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets_prep).sum().item()

            # END loop minibatches
            self.network.eval()
            test_loss = 0
            correct = 0
            total = 0
            # count = 0
            for inputs, targets_prep in self.dataset.minibatches(train=False):
                # count += 1

                targets = np.zeros((inputs.shape[0], 100), np.float32)
                targets[range(len(targets_prep)), targets_prep.type(torch.int32)] = 1.

                inputs = inputs.to(self.device)

                outputs = self.network.forward(inputs)  # make the embedding
                outputs = self.network.predict(outputs)  # make the prediction with sigmoid, making g_y(xi)

                targets = torch.tensor(targets).to(outputs.device)
                loss_bx = self.loss(outputs, targets)
                test_loss += loss_bx.item()

                targets_prep = torch.LongTensor(targets_prep).to(outputs.device)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets_prep).sum().item()

                total += targets.size(0)

            acc = 100. * correct / total
            if (epoch + 1) % 7 == 0:
                print(f"Epoch {epoch + 1} : Loss {test_loss / total:.8f} - Accuracy {acc:.2f}")

                # adjust learning rate
            if (epoch + 1) in self.lr_strat:
                new_lr = new_lr / self.lr_factor
                print("New LR:" + str(new_lr))
                optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.network.parameters()), lr=new_lr, momentum=0.9,
                                      weight_decay=self.decay, nesterov=False)

        # Duplicate current network to distillate info
        self.network2 = copy.deepcopy(self.network)
        self.network2.eval()

    def update_exemplars(self, iteration):
        nb_protos_cl = int(np.ceil(self.mem_size / self.nb_cl / (iteration + 1)))  # num of exemplars per class
        # Herding
        self.network.eval()

        for iter_dico in range(self.nb_cl):
            cl = self.dataset.order[iteration * self.nb_cl + iter_dico]

            # Possible exemplars in the feature space and projected on the L2 sphere
            # exemplars of class iter_dico + nb_cl
            pinput = tensor(self.dataset.get_X_of_class(cl)).to(self.device)
            mapped_prototypes = self.network.forward(pinput).cpu().detach().numpy()
            D = mapped_prototypes.T
            D = D / np.linalg.norm(D, axis=0)
            # Herding procedure : ranking of the potential exemplars
            mu = np.mean(D, axis=1)

            # set exemplar to zero
            self.alpha_dr_herding[cl, :] = self.alpha_dr_herding[cl, :] * 0
            w_t = mu
            iter_herding = 0
            iter_herding_eff = 0
            # Herding algorithm
            while not (np.sum(self.alpha_dr_herding[cl, :] != 0) == min(nb_protos_cl, 500)) and iter_herding_eff < 1000:
                tmp_t = np.dot(w_t, D)
                ind_max = np.argmax(tmp_t)
                iter_herding_eff += 1
                if self.alpha_dr_herding[cl, ind_max] == 0:
                    self.alpha_dr_herding[cl, ind_max] = 1 + iter_herding
                    iter_herding += 1
                w_t = w_t + mu - D[:, ind_max]

        # Prepare the protoset
        X_protoset_cumuls = []
        Y_protoset_cumuls = []

        # Storing the selected exemplars in the protoset
        for iteration2 in range(iteration + 1):

            for iter_dico in range(self.nb_cl):
                cl = self.dataset.order[iteration2 * self.nb_cl + iter_dico]

                alph = self.alpha_dr_herding[cl, :]  # select the herd of the current class
                alph = (alph > 0) * (alph < nb_protos_cl + 1) * 1.  # put one in the ones to select

                # append exeplars in the protoset
                X_protoset_cumuls.append(self.dataset.get_X_of_class(cl)[np.where(alph == 1)[0]])
                Y_protoset_cumuls.append(cl * np.ones(len(np.where(alph == 1)[0]), dtype=np.int32))

        return X_protoset_cumuls, Y_protoset_cumuls

    def compute_means(self, iteration=10):

        class_means = np.zeros((64, 100, 2))
        nb_protos_cl = int(np.ceil(self.mem_size / self.nb_cl / iteration))  # num of exemplars per class

        for iteration2 in range(iteration):

            for iter_dico in range(self.nb_cl):
                cl = self.dataset.order[iteration2 * self.nb_cl + iter_dico]
                pinput = tensor(self.dataset.get_X_of_class(cl)).to(self.device)

                # Collect data in the feature space for each class
                mapped_prototypes = self.network.forward(pinput).cpu().detach().numpy()
                D = mapped_prototypes.T
                D = D / np.linalg.norm(D, axis=0)

                # Flipped version also # Check performance, se uguali levalo
                # inverted = np.array(self.dataset.get_X_of_class(cl)[:, :, :, ::-1])
                # pinput2 = tensor(np.array((inverted - self.dataset.pixel_means), dtype=np.float32)).to(self.device)
                # mapped_prototypes2 = self.network.forward(pinput2).cpu().detach().numpy()
                # D2 = mapped_prototypes2.T
                # D2 = D2 / np.linalg.norm(D2, axis=0)

                # iCaRL
                alph = self.alpha_dr_herding[cl, :]  # importance of each image of this class
                alph = (alph > 0) * (alph < nb_protos_cl + 1) * 1.  # 1 if in the current herd

                alph = alph / np.sum(alph)  # to make the average only for the current prototypes.
                class_means[:, cl, 0] = (np.dot(D, alph))  # + np.dot(D2, alph)) / 2
                # dot operation is for weighting each f(xi) with alpha
                class_means[:, cl, 0] /= np.linalg.norm(class_means[:, cl, 0])

                # Normal NCM
                alph = np.ones(self.dictionary_size) / self.dictionary_size  # to make the avg over all samples
                class_means[:, cl, 1] = (np.dot(D, alph))  # + np.dot(D2, alph)) / 2
                # dot operation is for weighting each f(xi) with alpha
                class_means[:, cl, 1] /= np.linalg.norm(class_means[:, cl, 1])

        np.save('cl_means', class_means)
        return class_means

    def test(self, iteration, class_means=None):

        if class_means is None:
            class_means = self.compute_means(iteration)

        top1_acc_list = np.zeros(3)

        stat_hb1 = []
        stat_icarl = []
        stat_ncm = []

        for inputs, targets_prep in self.dataset.minibatches_for_test(iteration):
            inputs = inputs.to(self.device)

            # compute prediction
            outputs = self.network.forward(inputs)  # returns embeddings
            pred = self.network.predict(outputs).cpu().detach().numpy()  # return classes of Hybrid1
            outputs = outputs.cpu().detach().numpy()

            outputs = (outputs.T / np.linalg.norm(outputs.T, axis=0)).T  # normalize output

            # Compute score for iCaRL
            sqd = cdist(class_means[:, :, 0].T, outputs, 'sqeuclidean')  # Squared euclidean distance
            score_icarl = (-sqd).T
            # Compute score for NCM
            sqd = cdist(class_means[:, :, 1].T, outputs, 'sqeuclidean')  # Squared euclidean distance
            score_ncm = (-sqd).T

            # Compute the accuracy over the batch
            targets_prep = targets_prep.numpy()

            stat_hb1 += ([ll in best for ll, best in zip(targets_prep, np.argsort(pred, axis=1)[:, -1:])])
            stat_icarl += ([ll in best for ll, best in zip(targets_prep, np.argsort(score_icarl, axis=1)[:, -1:])])
            stat_ncm += ([ll in best for ll, best in zip(targets_prep, np.argsort(score_ncm, axis=1)[:, -1:])])

        top1_acc_list[0] = np.average(stat_icarl) * 100  # ICarl
        top1_acc_list[1] = np.average(stat_hb1) * 100  # Hybrid 1
        top1_acc_list[2] = np.average(stat_ncm) * 100  # NCM

        return top1_acc_list
