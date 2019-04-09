import torch
import torch.nn as nn
import torch.optim as optim
import networks.networks as net
import torchvision as tv
from torchvision import transforms
from torch.utils.data import DataLoader
from data.idadataloader import DoubleDataset
import torch.nn.functional as F
import numpy as np

root = '/home/fcdl/dataset/'
target_path = root + "GTSRB/Final_Training"
source_path = root + "synthetic_data"
test_path = root + "GTSRB/Final_Test"

EPOCHS = 70
NUM_CLASSES = 43
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class EntropyLoss(nn.Module):
    ''' Module to compute entropy loss '''
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(-1).mean()
        return b


def update_stats(network, train_loader, scheduler=None, optimizer=None):
    network.train()
    for batch in train_loader:

        # optimizer.zero_grad()

        # train the source
        network.set_target()

        inputs, targets = batch

        inputs = inputs.to(device)
        #targets = targets.to(device)

        outputs = network.forward(inputs)  # feature vector only
        #prediction = network.predict(outputs)  # make the prediction with sigmoid, making g_y(xi)
    print("Stats updated")
    return 0., 0.


def train_epoch_single(network, train_loader, scheduler, optimizer):
    # src_criterion = nn.CrossEntropyLoss()
    src_criterion = nn.BCEWithLogitsLoss(reduction='mean')

    network.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    batch_idx = 0
    scheduler.step()

    for batch in train_loader:

        optimizer.zero_grad()

        # train the source
        network.set_source()

        inputs, targets = batch

        inputs = inputs.to(device)

        targets_bce = np.zeros((inputs.shape[0], NUM_CLASSES), np.float32)
        targets_bce[range(len(targets)), targets.type(torch.int32)] = 1.

        targets = targets.to(device)
        targets_bce = torch.tensor(targets_bce).to(device)

        outputs = network.forward(inputs)  # feature vector only
        prediction = network.predict(outputs)  # make the prediction with sigmoid, making g_y(xi)

        loss_bx = src_criterion(prediction, targets_bce)  # CE loss

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


def train_epoch(network, train_loader, scheduler, optimizer):
    tar_criterion = nn.CrossEntropyLoss()
    src_criterion = nn.CrossEntropyLoss()

    network.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    train_correct_src = 0
    train_total_src = 0
    batch_idx = 0
    scheduler.step()

    # balance_factor = len(train_loader.dataset.dataset2) / len(train_loader.dataset.dataset1)
    print(f"{len(train_loader.dataset.dataset2)} / {len(train_loader.dataset.dataset1)}")

    for source_loader, target_loader in train_loader:

        optimizer.zero_grad()

        # train the source
        network.set_source()

        inputs, targets = source_loader

        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = network.forward(inputs)  # feature vector only
        prediction = network.predict(outputs)  # make the prediction with sigmoid, making g_y(xi)

        loss_bx_src = src_criterion(prediction, targets)  # CE loss

        # stats on target
        _, predicted = prediction.max(1)
        tr_tot = targets.size(0)
        tr_crc = predicted.eq(targets).sum().item()

        train_total_src += tr_tot
        train_correct_src += tr_crc

        # train the target
        network.set_target()

        inputs, targets = target_loader

        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = network.forward(inputs)  # feature vector only
        prediction = network.predict(outputs)  # make the prediction with sigmoid, making g_y(xi)

        loss_bx_tar = tar_criterion(prediction, targets)  # C-Entropy LOSS

        # stats on target
        _, predicted = prediction.max(1)
        tr_tot = targets.size(0)
        tr_crc = predicted.eq(targets).sum().item()
        train_total += tr_tot
        train_correct += tr_crc

        # sum the losses and do backward propagation
        loss_bx = loss_bx_src + loss_bx_tar

        loss_bx.backward()
        optimizer.step()

        # compute statistics
        train_loss += loss_bx.item()

        batch_idx += 1
        if batch_idx % 200 == 0:
            print(f"{batch_idx:3d} | "
                  f"Source Loss: {loss_bx_src:.6f} "
                  f"Source Acc : {100.0 * train_correct_src / train_total_src:.2f}"
                  f"    | "
                  f"Target Loss: {loss_bx_tar:.6f} "
                  f"Target Acc : {100.0 * train_correct / train_total:.2f}")

    train_acc = 100. * train_correct / train_total

    return train_loss/batch_idx, train_acc


def valid(network, valid_loader, target=True):
    criterion = nn.CrossEntropyLoss()
    # make validation
    network.eval()
    if target:
        network.set_target()
    else:
        network.set_source()
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
    source_loader = DataLoader(source, 128, True, num_workers=8)
    target_loader = DataLoader(target, 128, True, num_workers=8)

    test_loader = DataLoader(test, 512, False, num_workers=8)

    # get network
    #net = net.cifar_resnet_dial(None, NUM_CLASSES).to(device)
    net = net.cifar_resnet(None, NUM_CLASSES).to(device)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=2., weight_decay=1e-5, momentum=0.9, nesterov=False)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [int(0.7*EPOCHS), int(0.9*EPOCHS)], gamma=0.2)

    print("STARTING ....")
    # define training steps
    for epoch in range(20):
        train_loss, train_acc = train_epoch_single(net, train_loader=target_loader, optimizer=optimizer, scheduler=scheduler)
        val_loss, val_acc = valid(net, valid_loader=test_loader, target=False)
        print(f"Epoch {epoch+1:03d}: Train Loss {train_loss:.6f}, Train Acc {train_acc:.2f}\n"
              f"         : Valid Loss {val_loss:.6f}, Valid Acc {val_acc:.2f}")

    for epoch in range(0):
        # train epoch
        train_loss, train_acc = train_epoch(net, train_loader=train_loader, optimizer=optimizer, scheduler=scheduler)
        # train_loss, train_acc = train_epoch_single(net, train_loader=source_loader, optimizer=optimizer, scheduler=scheduler)
        # train_loss, train_acc = update_stats(net, train_loader=target_loader, optimizer=None, scheduler=None)
        # train_loss, train_acc = 0., 0.
        # valid!
        #val_loss, val_acc = valid(net, valid_loader=test_loader, target=False)
        val_loss, val_acc = valid(net, valid_loader=test_loader)

        print(f"Epoch {epoch+1:03d}: Train Loss {train_loss:.6f}, Train Acc {train_acc:.2f}\n"
              f"         : Valid Loss {val_loss:.6f}, Valid Acc {val_acc:.2f}")

    print("SAVING...")
    torch.save({
        "network": net.state_dict()
    }, "models/cifar_resnet_TO.pth")

    print(".... END")
