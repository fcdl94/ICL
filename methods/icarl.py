from torch import tensor
import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from datetime import datetime
from scipy.spatial.distance import cdist
from .common import *

LR = 2.
MEM_SIZE = 20 * 100
DECAY = 0.00001
EPOCHS = 70
LR_FACTOR = 5.
METHODS = ["iCaRL", "Hybrid", "NCM", "iCaRL-INV"]

DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'


class ICarl(AbstractMethod):

    def __init__(self, network, n_classes=100, nb_base=10, nb_incr=10,
                 mem_size=MEM_SIZE, distillation=True,
                 lr_init=LR, decay=DECAY, epochs=EPOCHS, device=DEVICE, log="ICARL"):
        """
        :param network: the backbone neural network of the model
        :param n_classes: Number of classes of the dataset
        :param mem_size: Number of prototypes to be stored
        :param lr_init: Initial learning rate (decayed by 5 after 0.7 and 0.9 epochs
        :param decay: weight decay for SGD
        :param epochs: number of epochs to train the model
        :param device: "cuda" or "cpu"
        """
        super().__init__(network=network, n_classes=n_classes, nb_base=nb_base, nb_incr=nb_incr, log=log)

        self.network = network.to(device)
        self.network2 = self.network

        self.loss = nn.BCEWithLogitsLoss(reduction='mean')

        self.dataset = None

        # method parameters
        self.mem_size = mem_size
        self.distillation = distillation
        self.prototypes = None
        self.alpha_dr_herding = [np.array([0]) for i in range(n_classes)]

        # training parameters
        self.device = device
        self.lr_init = lr_init
        self.decay = decay
        self.epochs = epochs
        self.lr_strat = [round(epochs * 0.7), round(epochs * 0.9)]
        self.lr_factor = LR_FACTOR

    def fit(self, dataset, checkpoint=None, epochs=None):

        self.alpha_dr_herding = [np.array([0]) for i in range(self.n_classes)]
        self.dataset = dataset

        if epochs is not None:
            self.epochs = epochs

        x_protoset = None
        y_protoset = None

        cumulative_accuracies = []

        start_iter = 0

        if checkpoint is not None:
            check_dict = torch.load(checkpoint)
            start_iter = check_dict['iteration']
            self.network.load_state_dict(check_dict['network'])
            self.network2.load_state_dict(check_dict['network2'])
            x_protoset = check_dict['X']
            y_protoset = check_dict['Y']

        for iteration in range(start_iter, self.iteration_total):

            # Prepare the training data for the current batch of classes
            data_loader = dataset.next_iteration(x_protoset, y_protoset)

            # TRAIN THIS ITERATION #
            print('Batch of classes number {0} arrives ...'.format(iteration + 1))

            self.incremental_fit(iteration, data_loader)  # train for N epochs (after each epoch validate)

            # END OF TRAINING FOR THIS ITERATION #

            # UPDATE EXEMPLARS #
            print('Updating exemplar set...')
            x_protoset, y_protoset = self.update_exemplars(iteration)

            # Save training checkpoint
            # torch.save({
            #    'iteration': iteration,
            #    'network': self.network.state_dict(),
            #    'network2': self.network2.state_dict(),
            #    'X': x_protoset,
            #    'Y': y_protoset
            # }, "checkpoint/iter_" + str(iteration) + "_checkpoint.pth.tar")

            # COMPUTE ACCURACY ##
            means = self.compute_means(iteration)
            acc_cum = self.test(iteration, class_means=means)
            acc_new = self.test(iteration, class_means=means, cumulative=False)
            acc_base = self.test(0, class_means=means)

            print_accuracy(METHODS, acc_base, acc_new, acc_cum)

            cumulative_accuracies.append(acc_cum)

            for i, name in enumerate(METHODS):
                save_results(f"{self.log_folder}/{name}.csv",
                             acc_base[i], acc_new[i], acc_cum[i])

        acc_cum = []
        tot = self.iteration_total - 1
        means = self.compute_means(tot)
        for i in range(tot+1):
            acc_cum.append(self.test(i, cumulative=False, class_means=means))

        save_per_batch_result(f"{self.log_folder}/per-batch.csv",
                                           ["iCaRL", "Hybrid", "NCM", "iCaRL-INV"], acc_cum)

        torch.save(
            {
                "network": self.network.state_dict()
            },
            f"models/icarl{datetime.now().isoformat()}.pth"
        )

        return cumulative_accuracies

    def compute_num_classes(self, iteration):
        if iteration < 0:
            return 0
        return self.nb_base + self.nb_incr * iteration

    def nb_classes(self, iteration):
        if iteration > 0:
            nb_cl = self.nb_incr
        else:
            nb_cl = self.nb_base
        return nb_cl

    def incremental_fit(self, iteration, data_loader):
        new_lr = self.lr_init
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.network.parameters()), lr=new_lr, momentum=0.9,
                              weight_decay=self.decay, nesterov=False)

        for epoch in range(self.epochs):
            self.network.train()
            train_loss = 0
            correct = 0
            total = 0

            # In each epoch, we do a full pass over the training data:
            for inputs, targets_prep in data_loader:

                targets = np.zeros((inputs.shape[0], self.n_classes), np.float32)  # 100 = classes of cifar
                targets[range(len(targets_prep)), targets_prep.type(torch.int32)] = 1.  # prepare target for CE loss

                inputs = inputs.to(self.device)

                optimizer.zero_grad()
                outputs = self.network.forward(inputs)  # feature vector only
                prediction = self.network.predict(outputs)  # make the prediction with sigmoid, making g_y(xi)
                targets = tensor(targets).to(outputs.device)
                targets_prep = torch.LongTensor(targets_prep).to(outputs.device)

                if iteration > 0 and self.distillation:  # apply distillation
                    outputs_old = self.network2.forward(inputs)
                    prediction_old = self.network2.predict(outputs_old)
                    to = self.compute_num_classes(iteration-1)  # until the number of classes of last iteration
                    targets[:, np.array(self.dataset.order[range(0, to)])] = \
                        torch.sigmoid(prediction_old[:, np.array(self.dataset.order[range(0, to)])])

                loss_bx = self.loss(prediction, targets)  # joins classification and distillation losses
                loss_bx.backward()
                optimizer.step()

                train_loss += loss_bx.item()
                _, predicted = prediction.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets_prep).sum().item()

            # if VALIDATION:
            # self.network.eval()
            # test_loss = 0
            # correct = 0
            # total = 0
            # # count = 0
            # for inputs, targets_prep in self.dataset.minibatches(train=False):
            #     # count += 1
            #
            #     targets = np.zeros((inputs.shape[0], 100), np.float32)
            #     targets[range(len(targets_prep)), targets_prep.type(torch.int32)] = 1.
            #
            #     inputs = inputs.to(self.device)
            #
            #     outputs = self.network.forward(inputs)  # make the embedding
            #     outputs = self.network.predict(outputs)  # make the prediction with sigmoid, making g_y(xi)
            #
            #     targets = torch.tensor(targets).to(outputs.device)
            #     loss_bx = self.loss(outputs, targets)
            #     test_loss += loss_bx.item()
            #
            #     targets_prep = torch.LongTensor(targets_prep).to(outputs.device)
            #     _, predicted = outputs.max(1)
            #     correct += predicted.eq(targets_prep).sum().item()
            #
            #     total += targets.size(0)
            #

            acc = 100. * correct / total

            print(f"Epoch {epoch + 1:3d} : Train Loss {train_loss / len(data_loader):.6f}, Train Acc {acc:.2f}")

            # adjust learning rate
            if (epoch + 1) in self.lr_strat:
                new_lr = new_lr / self.lr_factor
                print("New LR:" + str(new_lr))
                optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.network.parameters()), lr=new_lr,
                                      momentum=0.9, weight_decay=self.decay, nesterov=False)

        # Duplicate current network to distillate info
        self.network2 = copy.deepcopy(self.network)
        self.network2.eval()

    def update_exemplars(self, iteration):
        nb_protos_cl = self.mem_size // self.compute_num_classes(iteration)  # num of exemplars per class
        # Herding
        self.network.eval()

        # Prepare the protoset
        x_protoset = None
        y_protoset = None

        if nb_protos_cl > 0:

            for iter_dico in range(self.nb_classes(iteration)):
                cl = self.dataset.order[self.compute_num_classes(iteration-1) + iter_dico].item()

                # per controllare che sia in ordine, qui si potrebbe prendere classi e mapparle
                pinput = self.dataset.get_dataloader_of_class(cl)

                output = []
                for img, tar in pinput:
                    img = img.to(self.device)
                    output.append(self.network.forward(img).cpu().detach())

                # Collect data in the feature space for each class
                mapped_prototypes = torch.cat(output).numpy()
                D = mapped_prototypes.T
                D = D / np.linalg.norm(D, axis=0)
                # Herding procedure : ranking of the potential exemplars
                mu = np.mean(D, axis=1)

                # set exemplar to zero
                self.alpha_dr_herding[cl] = np.zeros(mapped_prototypes.shape[0])  # number of rows
                w_t = mu
                iter_herding = 0
                iter_herding_eff = 0
                # Herding algorithm
                while not (np.sum(self.alpha_dr_herding[cl] != 0) == min(nb_protos_cl, mapped_prototypes.shape[0])) \
                        and iter_herding_eff < 1000:
                    tmp_t = np.dot(w_t, D)
                    ind_max = np.argmax(tmp_t)
                    iter_herding_eff += 1
                    if self.alpha_dr_herding[cl][ind_max] == 0:
                        self.alpha_dr_herding[cl][ind_max] = 1 + iter_herding
                        iter_herding += 1
                    w_t = w_t + mu - D[:, ind_max]

            x_protoset = []
            y_protoset = []

            # Storing the selected exemplars in the protoset
            for iteration2 in range(iteration + 1):

                for iter_dico in range(self.nb_classes(iteration2)):
                    # pick actual class
                    cl = self.dataset.order[self.compute_num_classes(iteration2-1) + iter_dico].item()

                    alph = self.alpha_dr_herding[cl]  # select the herd of the current class
                    alph = (alph > 0) * (alph < nb_protos_cl + 1) * 1.  # put one in the ones to select

                    # append exeplars in the protoset
                    img = self.dataset.get_images_of_class(cl)
                    x_protoset += [img[j] for j in range(len(alph)) if alph[j] == 1]
                    y_protoset += [cl for j in range(len(alph)) if alph[j] == 1]

        return x_protoset, y_protoset

    def compute_means(self, iteration):

        class_means = np.zeros((64, 100, 3))
        nb_protos_cl = self.mem_size // self.compute_num_classes(iteration)  # num of exemplars per class

        if nb_protos_cl > 0:
            for iteration2 in range(iteration + 1):

                for iter_dico in range(self.nb_classes(iteration2)):
                    # pick actual class
                    cl = self.dataset.order[self.compute_num_classes(iteration2 - 1) + iter_dico].item()

                    # compute network resposes for images of class cl
                    pinput = self.dataset.get_dataloader_of_class(cl)
                    output = []
                    for img, tar in pinput:
                        img = img.to(self.device)
                        output.append(self.network.forward(img).cpu().detach())

                    # Collect data in the feature space for each class
                    mapped_prototypes = torch.cat(output).numpy()  # should be 500 x 64 in CIFAR
                    D = mapped_prototypes.T  # now each column is a sample # 64 x 500
                    D = D / np.linalg.norm(D, axis=0)  # 64 x 500

                    # compute network resposes for images of class cl for flipped images
                    flip = transforms.RandomHorizontalFlip(p=1)
                    pinput = self.dataset.get_dataloader_of_class(cl, flip)
                    output = []
                    for img, tar in pinput:
                        img = img.to(self.device)
                        output.append(self.network.forward(img).cpu().detach())

                    mapped_prototypes_flip = torch.cat(output).numpy()  # should be 500 x 64 in CIFAR
                    D2 = mapped_prototypes_flip.T  # now each column is a sample # 64 x 500
                    D2 = D2 / np.linalg.norm(D2, axis=0)  # 64 x 500

                    # iCaRL
                    alph = self.alpha_dr_herding[cl]  # importance of each image of this class
                    dict_size = len(self.alpha_dr_herding[cl])
                    alph = (alph > 0) * (alph < nb_protos_cl + 1) * 1.  # 1 if in the current herd

                    # Handle the case in which there are no prototypes
                    s = np.sum(alph)
                    if s == 0:
                        s = 1

                    alph = alph / s  # to make the average only for the current prototypes.
                    class_means[:, cl, 0] = (np.dot(D, alph))
                    # dot operation is for weighting each f(xi) with alpha
                    class_means[:, cl, 0] /= np.linalg.norm(class_means[:, cl, 0])

                    # Inverted ICaRL
                    class_means[:, cl, 2] = (np.dot(D, alph) + np.dot(D2, alph)) / 2
                    # dot operation is for weighting each f(xi) with alpha
                    class_means[:, cl, 2] /= np.linalg.norm(class_means[:, cl, 2])

                    # Normal NCM
                    alph = np.ones(dict_size) / dict_size  # to make the avg over all samples
                    class_means[:, cl, 1] = (np.dot(D, alph))
                    # dot operation is for weighting each f(xi) with alpha
                    class_means[:, cl, 1] /= np.linalg.norm(class_means[:, cl, 1])

        return class_means

    def test(self, iteration, cumulative=True, class_means=None):

        if class_means is None and self.mem_size > 0:
            class_means = self.compute_means(iteration)

        top1_acc_list = np.zeros(4)

        stat_hb1 = []
        stat_icarl = []
        stat_ncm = []
        stat_icarl_i = []

        data_loader = self.dataset.test_dataloader(iteration, cumulative=cumulative)

        for inputs, targets_prep in data_loader:
            inputs = inputs.to(self.device)

            # compute prediction
            outputs = self.network.forward(inputs)  # returns embeddings
            pred = self.network.predict(outputs).cpu().detach().numpy()  # return score classes as logits
            outputs = outputs.cpu().detach().numpy()

            outputs = (outputs.T / np.linalg.norm(outputs.T, axis=0)).T  # normalize output

            # Compute the accuracy over the batch
            targets_prep = targets_prep.numpy()

            if self.mem_size > 0:
                # Compute score for iCaRL
                sqd = cdist(class_means[:, :, 0].T, outputs, 'sqeuclidean')  # Squared euclidean distance
                score_icarl = (-sqd).T
                # Compute score for iCaRL-Inverted
                sqd = cdist(class_means[:, :, 2].T, outputs, 'sqeuclidean')  # Squared euclidean distance
                score_icarl_inv = (-sqd).T
                # Compute score for NCM
                sqd = cdist(class_means[:, :, 1].T, outputs, 'sqeuclidean')  # Squared euclidean distance
                score_ncm = (-sqd).T

                stat_icarl += ([ll in best for ll, best in zip(targets_prep, np.argsort(score_icarl, axis=1)[:, -1:])])
                stat_icarl_i += (
                [ll in best for ll, best in zip(targets_prep, np.argsort(score_icarl_inv, axis=1)[:, -1:])])
                stat_ncm += ([ll in best for ll, best in zip(targets_prep, np.argsort(score_ncm, axis=1)[:, -1:])])

            stat_hb1 += ([ll in best for ll, best in zip(targets_prep, np.argsort(pred, axis=1)[:, -1:])])
            # use the logits

        if self.mem_size > 0:
            top1_acc_list[0] = np.average(stat_icarl) * 100.  # ICarl
            top1_acc_list[2] = np.average(stat_ncm) * 100.  # NCM
            top1_acc_list[3] = np.average(stat_icarl_i) * 100.  # ICaRL inv

        top1_acc_list[1] = np.average(stat_hb1) * 100.  # Hybrid 1

        return top1_acc_list

    def predict(self, inputs, method=0):
        """
        :return the predicted class for the inputs as tensor in cpu

        :param inputs: tensors to be evaluated
        :param method: 0 (def) is ICaRL, 1 is NCM, 2 is ICaRL-inv, 3 is with sigmoid (Hybrid 1)
        """

        if self.iteration_total > 0 and 0 <= method <= 3:

            class_means = self.compute_means(self.iteration_total-1)
            outputs = self.network.forward(inputs)  # returns embeddings

            if method == 3:
                pred = self.network.predict(outputs).cpu().detach()
            elif 0 <= method < 3:
                outputs = outputs.cpu().detach().numpy()
                outputs = (outputs.T / np.linalg.norm(outputs.T, axis=0)).T  # normalize output
                sqd = cdist(class_means[:, :, method].T, outputs, 'sqeuclidean')  # Squared euclidean distance
                score = (-sqd).T
                pred = np.argsort(score, axis=1)[:, -1:]
                pred = torch.tensor(pred)

            return pred
        else:
            raise Exception("Pass method between 0 and 3 inclusive")
