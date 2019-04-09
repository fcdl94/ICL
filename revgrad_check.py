import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import networks.networks as net
import torchvision as tv
from torchvision import transforms
from torch.utils.data import DataLoader
from data.idadataloader import DoubleDataset
import torch.nn.functional as F

root = '/home/fcdl/dataset/'
target_path = root + "GTSRB/Final_Training"
source_path = root + "synthetic_data"
test_path = root + "GTSRB/Final_Test"

EPOCHS = 70
NUM_CLASSES = 43
device = 'cuda' if torch.cuda.is_available() else 'cpu'
const = 1


def train_epoch(network, train_loader, scheduler, optimizer):
    src_criterion = nn.CrossEntropyLoss()
    dom_criterion = nn.BCEWithLogitsLoss()

    network.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    batch_idx = 0
    scheduler.step()

    for source_loader, target_loader in train_loader:

        p = float(batch_idx + start_steps) / total_steps
        lam = 2. / (1. + np.exp(-10 * p)) - 1

        optimizer.zero_grad()

        # train the source
        network.set_source()

        inputs, targets = source_loader

        inputs = inputs.to(device)
        targets = targets.to(device)
        domains = torch.zeros(inputs.shape[0], 1).to(device)

        outputs = network.forward(inputs)  # feature vector only
        prediction = network.predict(outputs)  # make the prediction with sigmoid, making g_y(xi)
        d_prediction = network.discriminate_domain(outputs)

        loss_bx_src = src_criterion(prediction, targets)  # CE loss
        loss_bx_dom_s = dom_criterion(d_prediction, domains)

        # train the target
        network.set_target()

        inputs, targets = target_loader

        inputs, targets = inputs.to(device), targets.to(device)
        domains = torch.ones(inputs.shape[0], 1).to(device)

        outputs = network.forward(inputs)  # feature vector only
        prediction = network.predict(outputs)
        d_prediction = network.discriminate_domain(outputs)  # make the prediction with sigmoid, making g_y(xi)

        loss_bx_tar = src_criterion(prediction, targets)
        loss_bx_dom_t = dom_criterion(d_prediction, domains)

        # sum the losses and do backward propagation
        loss_dom = (loss_bx_dom_s + loss_bx_dom_t)
        #loss_bx = loss_bx_src + loss_bx_tar + const * lam * loss_dom
        loss_bx = loss_bx_src + const * lam * loss_dom

        loss_bx.backward()
        optimizer.step()

        _, predicted = prediction.max(1)
        tr_tot = targets.size(0)
        tr_crc = predicted.eq(targets).sum().item()

        # compute statistics
        train_loss += loss_bx.item()
        train_total += tr_tot
        train_correct += tr_crc

        batch_idx += 1
        if batch_idx % 200 == 0:
            print(f"Lambda {lam:.4f} "
                  f"Domain Loss: {loss_dom:.6f} "
                  f"Source Loss: {loss_bx_src:.6f} "
                  f"Target Loss: {loss_bx_tar:.6f}"
                  f"Target Acc : {100.0 * train_correct / train_total:.2f}")

    train_acc = 100. * train_correct / train_total

    return train_loss/batch_idx, train_acc


def valid(network, valid_loader):
    criterion = nn.CrossEntropyLoss()
    # make validation
    network.eval()
    network.set_target()
    test_loss = 0
    test_correct = 0
    test_total = 0
    for inputs, targets in valid_loader:

        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = network.forward(inputs)  # make the embedding
        outputs = network.predict(outputs)  # make the prediction with sigmoid, making g_y(xi)

        loss_bx = criterion(outputs, targets)  # without distillation? -> YES, validation only on new classes

        test_loss += loss_bx.item()
        _, predicted = outputs.max(1)
        test_total += targets.size(0)
        test_correct += predicted.eq(targets).sum().item()

    # normalize and print stats
    test_acc = 100. * test_correct / test_total
    test_loss /= len(valid_loader)

    return test_loss, test_acc


if __name__ == '__main__':

    # define transform
    # Normalize to have range between -1,1 : (x - 0.5) * 2
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    # Create data augmentation transform
    augmentation = transforms.Compose([transforms.Resize((35, 35)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomCrop((32, 32)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                       ])

    target = tv.datasets.ImageFolder(target_path, transform=augmentation)
    source = tv.datasets.ImageFolder(source_path, transform=augmentation)
    test = tv.datasets.ImageFolder(test_path, transform=transform)

    train = DoubleDataset(source, target)

    train_loader = DataLoader(train, 64, True, num_workers=8)

    test_loader = DataLoader(test, 512, False, num_workers=8)

    # get network
    net = net.cifar_resnet_revgrad(None, NUM_CLASSES).to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.1)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [int(0.7*EPOCHS), int(0.9*EPOCHS)], gamma=0.1)

    total_steps = EPOCHS * len(train_loader)
    # define training steps
    for epoch in range(EPOCHS):
        # steps
        start_steps = epoch * len(train_loader)

        # train epoch
        train_loss, train_acc = train_epoch(net, train_loader=train_loader, optimizer=optimizer, scheduler=scheduler)
        # valid!
        val_loss, val_acc = valid(net, valid_loader=test_loader)

        print(f"Epoch {epoch+1:03d}: Train Loss {train_loss:.6f}, Train Acc {train_acc:.2f}\n"
              f"         : Valid Loss {val_loss:.6f}, Valid Acc {val_acc:.2f}")

    torch.save({
        "network": net
    }, "models/cifar_resnet_rev_grad_ce_target.pth")

    print(".... END")

