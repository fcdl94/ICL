import numpy as np
import torch.optim as optim
import networks.networks as net
from networks.gtsrb import *
from networks.svhn import *
import torchvision as tv
from torchvision import transforms
from torch.utils.data import DataLoader
from data.idadataloader import DoubleDataset
from config import get_transform
from data.mnist_m import MNISTM
import argparse

parser = argparse.ArgumentParser(description='Sanity Checks Only')
parser.add_argument('setting', default="SO", help='Setting to run (see config.py)')
args = parser.parse_args()


root = '/home/fcdl/dataset/'
#target_path = root + "GTSRB/Final_Training/Images"
#source_path = root + "synthetic_data"
#test_path = root + "GTSRB/Final_Test"

#target_path = root + 'sketchy/photo_train'
#source_path = root + 'sketchy/sketch'
#test_path = root + 'sketchy/photo_test'

EPOCHS = 40
NUM_CLASSES = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'
const = 1


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

    return train_loss/batch_idx, train_acc


def train_epoch(network, train_loader, optimizer):
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

        inputs, targets = inputs.to(device), targets.to(device) # class gt
        domains = torch.ones(inputs.shape[0], 1).to(device)  # target is index 1

        logits, feat = network.forward(inputs)  # feature vector only
        prediction = network.predict(logits)  # class scores
        d_prediction = network.discriminate_domain(feat, lam)  # domain score

        loss_bx_tar = src_criterion(prediction, targets)
        loss_bx_dom_t = dom_criterion(d_prediction, domains)

        # sum the losses and do backward propagation
        loss_dom = (loss_bx_dom_s + loss_bx_dom_t)
        #loss_bx = loss_bx_src + loss_bx_tar + const * lam * loss_dom  # using target labels
        loss_bx = loss_bx_src + const * loss_dom              # don't use target labels

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

    return train_loss/batch_idx, train_acc


def valid(network, valid_loader):
    criterion = nn.CrossEntropyLoss()
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


if __name__ == '__main__':

    # define transform
    transform, augmentation = get_transform('svhn')
    augmentation = transforms.Compose([augmentation, transform])
    print(transform, augmentation)

    # define dataset
    #target = tv.datasets.ImageFolder(target_path, transform=augmentation)
    #source = tv.datasets.ImageFolder(source_path, transform=augmentation)
    #test = tv.datasets.ImageFolder(test_path, transform=transform)

    source = tv.datasets.SVHN(root, transform=augmentation)
    target = tv.datasets.MNIST(root, transform=tv.transforms.Compose([tv.transforms.Grayscale(3), transform]))
    test = tv.datasets.MNIST(root, train=False, transform=tv.transforms.Compose([tv.transforms.Grayscale(3), transform]))

    #source = tv.datasets.MNIST(root, transform=tv.transforms.Compose([tv.transforms.Grayscale(3), transform]))
    #target = MNISTM(root, transform=transform)
    #test = MNISTM(root, train=False, transform=transform)

    train = DoubleDataset(source, target)

    # define dataloader
    train_loader = DataLoader(train, 128, True, num_workers=8)
    source_loader = DataLoader(source, 128, True, num_workers=8)
    target_loader = DataLoader(target, 128, True, num_workers=8)

    test_loader = DataLoader(test, 128, False, num_workers=8)

    # get network
    #net = net.cifar_resnet_revgrad(None, NUM_CLASSES).to(device)
    #net = GTSRB_net(43).to(device)
    net = SVHN_net(10).to(device)
    #net = net.wide_resnet_revgrad(None, 125).to(device)
    #net = net.resnet50(True, 125).to(device)
    #net = LeNet().to(device)

    #optimizer = optim.SGD(net.parameters(), lr=0.1)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [int(0.7*EPOCHS), int(0.9*EPOCHS)], gamma=0.1)

    total_steps = EPOCHS * len(train_loader)

    print("Do a validation before starting to check it is ok...")
    val_loss, val_acc, dom_acc = valid(net, valid_loader=test_loader)
    print(f"Epoch {-1:03d} : Test Loss {val_loss:.6f}, Test Acc {val_acc:.2f}, Domain Acc {dom_acc:.2f}")
    print("Result should be random guessing, i.e. 10% accuracy")

    # define training steps
    for epoch in range(EPOCHS):
        # steps
        start_steps = epoch * len(train_loader)

        # train epoch
        learning_rate = 0.01 / ((1 + 10 * (epoch)/EPOCHS)**0.75)
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
        # scheduler.step()
        print(f"Learning rate: {learning_rate}")

        if args.setting == 'SO':
            train_loss, train_acc = train_epoch_single(net, train_loader=source_loader, optimizer=optimizer)
        elif args.setting == 'TO':
            train_loss, train_acc = train_epoch_single(net, train_loader=target_loader, optimizer=optimizer)
        else:
            train_loss, train_acc = train_epoch(net, train_loader=train_loader, optimizer=optimizer)

        # valid!
        val_loss, val_acc, dom_acc = valid(net, valid_loader=test_loader)

        print(f"\nEpoch {epoch+1:03d} : Test Loss {val_loss:.6f}, Test Acc {val_acc:.2f}, Domain Acc {dom_acc:.2f}\n")

        if train_loss < 1e-4:
            break

    print(".... END")

