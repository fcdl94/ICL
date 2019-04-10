from torch import tensor
import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import copy
from scipy.spatial.distance import cdist
from .common import *
import logging

ALL = 1000000
LR = 2.
MEM_SIZE = 20 * 100
DECAY = 0.00001
EPOCHS = 70
LR_FACTOR = 5.
METHODS = ["iCaRL", "Hybrid", "NCM", "iCaRL-INV"]

DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'


class ICarlDA(AbstractMethod):

    def __init__(self, network, n_classes=100, n_base=10, n_incr=10,
                 target_mem_size=ALL, source_mem_size=MEM_SIZE, distillation=True, features=64,
                 lr_init=LR, decay=DECAY, epochs=EPOCHS, device=DEVICE,
                 protos=True, log="ICARL", name="ICARL", **trash):

        super().__init__(network=network, n_classes=n_classes, nb_base=n_base, nb_incr=n_incr,
                         log=log, name=name,
                         lr_init=lr_init, decay=decay)

        self.network = network.to(device)
        self.network2 = self.network

        self.loss = nn.BCEWithLogitsLoss(reduction='mean')

        self.dataset = None

        # method variables
        self.protos = protos
        if protos:
            self.mem_size_target = target_mem_size
            self.mem_size_source = source_mem_size
        else:
            self.mem_size_target = 0
            self.mem_size_source = 0

        self.prototypes_target = None  # list of target PIL images
        self.prototypes_source = None  # list of source PIL images

        self.alpha_dr_herding = [np.array([0]) for _ in range(n_classes)]

        # training parameters
        self.features = features
        self.distillation = distillation
        self.device = device
        self.lr_init = lr_init
        self.decay = decay
        self.epochs = epochs
        self.lr_factor = 1. / LR_FACTOR

    # STARTS THE TRAINING OF THE WHOLE METHOD, ON EVERY BATCH CLASS
    def fit(self, dataset, checkpoint=None, epochs=None):
        self.dataset = dataset
        if epochs is not None:
            self.epochs = epochs
        cumulative_accuracies = []

        for iteration in range(0, self.iteration_total):
            # first iteration on target data
            if iteration == 0:
                train_loader, valid_dataloader = dataset.next_iteration(iteration=iteration)
                logging.info(f"Target (samples) len {len(train_loader.dataset)}")
            else:
                train_loader, valid_dataloader = dataset.next_iteration(target_proto=self.prototypes_target,
                                                                        source_proto=self.prototypes_source,
                                                                        iteration=iteration)
                if self.protos:
                    logging.info(f"Source (samples) len {len(train_loader.dataset.dataset1)}")
                    logging.info(f"Target (protos)  len {len(train_loader.dataset.dataset2)}")

            logging.info(f'{"Source" if iteration>0 else "Target"} batch {iteration} samples arrives ...')
            self.incremental_fit(iteration, train_loader, valid_dataloader)

            if self.protos:
                logging.info('Updating exemplar set ...')
                if iteration == 0:
                    self.prototypes_target = self.update_exemplars(iteration)
                else:
                    self.prototypes_source = self.update_exemplars(iteration)
                logging.info('Computing class means ...')
                means = self.compute_means(iteration)
            else:
                means = None

            acc_cum = self.test(iteration, class_means=means, conf_matrix=True)
            acc_new = self.test(iteration, class_means=means, cumulative=False)
            acc_base = self.test(0, class_means=means)

            print_accuracy(METHODS, acc_base, acc_new, acc_cum)

            cumulative_accuracies.append(acc_cum)

            for i, name in enumerate(METHODS):
                save_results(f"{self.log_folder}/{name}.csv",
                             acc_base[i], acc_new[i], acc_cum[i])

        acc_cum = []
        tot = self.iteration_total - 1
        if self.protos:
            means = self.compute_means(tot)
        else:
            means = None

        for i in range(tot+1):  # compute per class batch results
            acc_cum.append(self.test(i, cumulative=False, class_means=means))

        save_per_batch_result(f"{self.log_folder}/per-batch.csv",
                              ["iCaRL", "Hybrid", "NCM", "iCaRL-INV"], acc_cum)

        torch.save(
            {
                "network": self.network.state_dict()
            },
            f"models/{self.name}.pth"
        )

        return cumulative_accuracies

    # TRAIN ON ONE CLASS BATCH
    def incremental_fit(self, iteration, train_loader, valid_loader):
        new_lr = self.lr_init
        # define optimizer SGD
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.network.parameters()), lr=new_lr, momentum=0.9,
                              weight_decay=self.decay, nesterov=False)
        steps = [round(int(self.epochs * 0.7)), round(int(self.epochs * 0.9))]
        scheduler = MultiStepLR(optimizer, steps, self.lr_factor)

        # for "epochs" epochs run observe on train and valid dataloader
        for epoch in range(self.epochs):
            # compute epoch
            train_loss, train_acc, test_loss, test_acc = \
                self.observe(epoch, iteration, train_loader, valid_loader, scheduler, optimizer)
            # update statistics
            self.logger.log_training(epoch, train_loss, train_acc, test_loss, test_acc, iteration)

        # Duplicate current network to distillate info
        self.network2 = copy.deepcopy(self.network)
        self.network2.eval()

    # CORE OF TRAINING FUNCTIONS -> OBSERVE WILL TRAIN ONE EPOCH AND PERFORM VALIDATION
    def observe(self, epoch, iteration, train_loader, valid_loader, scheduler, optimizer):
        # start with training
        self.network.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        scheduler.step()

        if iteration == 0 or not self.protos:  # if I DON'T use protos (I don't use them in first iteration as well)
            if iteration == 0:
                self.network.set_target()
            else:  # case of training without exemplars (e.g. LwF)
                self.network.set_source()
                self.network2.set_source()

            for batch in train_loader:
                optimizer.zero_grad()

                # compute loss
                loss_bx, trt_tot, trt_crc = self._compute_loss(batch, iteration)
                # perform backward propagation!
                loss_bx.backward()
                optimizer.step()
                # update stats
                train_loss += loss_bx.item()
                train_total += trt_tot
                train_correct += trt_crc
        else:  # if I USE protos
            for source_batch, target_batch in train_loader:
                optimizer.zero_grad()

                # train the source
                self.network.set_source()
                self.network2.set_source()
                loss_bx_src, tr_tot, tr_crc = self._compute_loss(source_batch, iteration)
                train_total += tr_tot
                train_correct += tr_crc

                # train the target
                self.network.set_target()
                self.network2.set_target()
                loss_bx_tar, tr_tot, tr_crc = self._compute_loss(target_batch, iteration)
                train_total += tr_tot
                train_correct += tr_crc

                # sum two losses and backprop!
                loss_bx = loss_bx_src + loss_bx_tar
                loss_bx.backward()
                optimizer.step()

                train_loss += loss_bx.item()

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
        train_loss /= len(train_loader)
        test_loss /= len(valid_loader)

        return train_loss, train_acc, test_loss, test_acc

    def _compute_loss(self, loader, iteration):
        inputs, targets_prep = loader

        targets = np.zeros((inputs.shape[0], self.n_classes), np.float32)
        targets[range(len(targets_prep)), targets_prep.type(torch.int32)] = 1.

        inputs = inputs.to(self.device)

        outputs = self.network.forward(inputs)  # feature vector only
        prediction = self.network.predict(outputs)  # make the prediction with sigmoid, making g_y(xi)
        targets = tensor(targets).to(outputs.device)
        targets_prep = torch.LongTensor(targets_prep).to(outputs.device)

        if iteration > 0 and self.distillation:  # apply distillation
            outputs_old = self.network2.forward(inputs)
            prediction_old = self.network2.predict(outputs_old)
            to = self.compute_num_classes(iteration - 1)  # until the number of classes of last iteration
            targets[:, np.array(self.dataset.order[range(0, to)])] = \
                torch.sigmoid(prediction_old[:, np.array(self.dataset.order[range(0, to)])])

        loss_bx = self.loss(prediction, targets)  # BCE joins classification and distillation losses
        _, predicted = prediction.max(1)
        train_total = targets.size(0)
        train_correct = predicted.eq(targets_prep).sum().item()

        return loss_bx, train_total, train_correct

    # TEST
    def test(self, iteration, cumulative=True, class_means=None, conf_matrix=False):

        top1_acc_list = np.zeros(4)

        stat_hb1 = []
        stat_icarl = []
        stat_ncm = []
        stat_icarl_i = []

        data_loader = self.dataset.test_dataloader(iteration, cumulative=cumulative)

        target_total = []
        target_pred = []

        self.network.set_target()
        self.network.eval()
        for inputs, targets_prep in data_loader:
            inputs = inputs.to(self.device)

            # compute prediction
            outputs = self.network.forward(inputs)  # returns embeddings
            pred = self.network.predict(outputs).cpu().detach().numpy()  # return score classes as logits
            outputs = outputs.cpu().detach().numpy()

            outputs = (outputs.T / np.linalg.norm(outputs.T, axis=0)).T  # normalize output

            # Compute the accuracy over the batch
            targets_prep = targets_prep.numpy()

            if class_means is not None:  # if we are using the means compute and use ICaRL
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
                stat_icarl_i += ([ll in best for ll, best in zip(targets_prep, np.argsort(score_icarl_inv, axis=1)[:, -1:])])
                stat_ncm += ([ll in best for ll, best in zip(targets_prep, np.argsort(score_ncm, axis=1)[:, -1:])])
                target_pred += [i.item() for i in np.argsort(score_icarl, axis=1)[:, -1:]]
            else:  # otherwise use H1 (special case to handle LwF)
                target_pred += [i.item() for i in np.argsort(pred, axis=1)[:, -1:]]

            stat_hb1 += ([ll in best for ll, best in zip(targets_prep, np.argsort(pred, axis=1)[:, -1:])])

            target_total += [i.item() for i in targets_prep]

        # use the logits
        top1_acc_list[0] = np.average(stat_icarl) * 100. if class_means is not None else 0.     # ICarl
        top1_acc_list[1] = np.average(stat_hb1) * 100.                                          # Hybrid 1
        top1_acc_list[2] = np.average(stat_ncm) * 100. if class_means is not None else 0.       # NCM
        top1_acc_list[3] = np.average(stat_icarl_i) * 100. if class_means is not None else 0.   # ICaRL inv
        # print confusion matrix
        if conf_matrix:
            self.logger.confusion_matrix(self.reorder_target(target_total),
                                         self.reorder_target(target_pred),
                                         self.n_classes)
        return top1_acc_list

    # EXEMPLARS SECTION -> UPDATE EXEMPLARS THAT UPDATE HERD (COMPUTING AND UPDATING IT)
    def update_exemplars(self, iteration):
        if iteration == 0:  # target case
            nb_protos_cl = self.mem_size_target // self.nb_base
            # print(f"Exemplar size target : {nb_protos_cl}")
            self.network.set_target()
        else:  # source case
            nb_protos_cl = self.mem_size_source // (self.compute_num_classes(iteration) - self.nb_base)
            # print(f"Exemplar size source : {nb_protos_cl}")
            self.network.set_source()

        # Herding
        self.network.eval()

        # Prepare the protoset
        x_protoset = None
        y_protoset = None

        if nb_protos_cl > 0:  # IF MEMORY FOR PROTOS IS > 0 (if I use them)
            # make the herd of new classes (storing the ordered queue in dataset)!
            for iter_dico in range(self.nb_classes(iteration)):  # iterate on the classes in current batch
                cl = self.dataset.order[self.compute_num_classes(iteration-1) + iter_dico].item()  # get class
                self._compute_herd(cl, nb_protos_cl)  # compute herd for that class

            x_protoset = []
            y_protoset = []

            # Storing the selected exemplars in the protoset
            if iteration > 0:
                for iteration2 in range(1, iteration + 1):  # for every class in source dataset
                    for iter_dico in range(self.nb_classes(iteration2)):  # iterate on every class-batch
                        # pick actual class
                        cl = self.dataset.order[self.compute_num_classes(iteration2-1) + iter_dico].item()
                        x_p, y_p = self._update_herd(cl, nb_protos_cl)
                        x_protoset += x_p
                        y_protoset += y_p

            else:
                for iter_dico in range(self.nb_base):  # iterate on every class-batch
                    # pick actual class
                    cl = self.dataset.order[iter_dico].item()
                    x_p, y_p = self._update_herd(cl, nb_protos_cl)
                    x_protoset += x_p
                    y_protoset += y_p

        return x_protoset, y_protoset

    def _compute_herd(self, cl, nb_protos_cl):
        # get a dataloader for current class (dataset already handle source and target distinction on cl)
        pinput = self.dataset.get_dataloader_of_class(cl)

        # compute output of classes
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
        while not (iter_herding == min(nb_protos_cl, mapped_prototypes.shape[0])) \
                and iter_herding_eff < 10000:
            tmp_t = np.dot(w_t, D)
            ind_max = np.argmax(tmp_t)
            iter_herding_eff += 1
            if self.alpha_dr_herding[cl][ind_max] == 0:
                self.alpha_dr_herding[cl][ind_max] = 1 + iter_herding
                iter_herding += 1
            w_t = w_t + mu - D[:, ind_max]

    def _update_herd(self, cl, nb_protos_cl):
        alph = self.alpha_dr_herding[cl]  # select the herd of the current class
        alph = (alph > 0) * (alph < nb_protos_cl + 1) * 1.  # put one in the exemplars to select

        # append exeplars in the protoset
        img = self.dataset.get_images_of_class(cl)  # get images in the order (handled by dl) of cl-dl
        x_protoset = [img[j] for j in range(len(alph)) if alph[j] == 1]
        y_protoset = [cl for j in range(len(alph)) if alph[j] == 1]
        return x_protoset, y_protoset

    # MEAN SECTION
    def compute_means(self, iteration):

        class_means = np.zeros((self.features, self.n_classes, 3))
        nb_protos_cl_target = self.mem_size_target // self.nb_base  # num of exemplars per base class

        if iteration == 0:
            nb_protos_cl_source = 0
        else:
            nb_protos_cl_source = self.mem_size_source // (self.compute_num_classes(iteration) - self.nb_base)

        self.network.set_target()
        for iter_dico in range(self.nb_classes(0)):
            # pick actual class
            cl = self.dataset.order[iter_dico].item()
            self._mean_of_class(class_means, cl, nb_protos_cl_target)

        self.network.set_source()
        for iteration2 in range(1, iteration + 1):
            for iter_dico in range(self.nb_incr):
                cl = self.dataset.order[self.compute_num_classes(iteration2 - 1) + iter_dico].item()
                self._mean_of_class(class_means, cl, nb_protos_cl_source)

        return class_means

    def _mean_of_class(self, class_means, cl, nb_protos_cl):

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

    # UTILS
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
