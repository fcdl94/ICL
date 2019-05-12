from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torchvision as tv
from torchvision import transforms
import datetime

from data import MNISTM
from data import DoubleDataset
from data.common import get_index_of_classes
from networks.svhn import lenet_net, svhn_net
from train import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--suffix', default="0", help='The suffix for the name of experiment')
parser.add_argument('-D', default=1, type=float)
parser.add_argument('-Y', default=0, type=float)
parser.add_argument('-T', default=0, type=float)
parser.add_argument('--revgrad', action='store_true')
parser.add_argument('--dataset', default="mnist")
parser.add_argument('--uda', action='store_true')
args = parser.parse_args()

# parameters and utils
device = 'cuda'
ROOT = '/home/fcdl/dataset/'
setting = f"{'uda' if args.uda else 'mixed'}-{args.dataset}"
method = 'dann' if args.revgrad else f'snnl-{args.D}-{args.T}'
method += f"_{args.suffix}"
save_name = f"models/{setting}/method.pth"


def get_setting():

    if args.dataset == 'mnist':
        transform = tv.transforms.Compose([transforms.Resize((28, 28)),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        source = tv.datasets.MNIST(ROOT, train=True, download=True,
                                   transform=tv.transforms.Compose([
                                       tv.transforms.Grayscale(3),
                                       transform])
                                   )
        test = MNISTM(ROOT, train=False, download=True, transform=transform)
        target = MNISTM(ROOT, train=True, download=True, transform=transform)
        EPOCHS = 40
        net = lenet_net().to(device)
    else:
        transform = tv.transforms.Compose([transforms.Resize((28, 28)),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        source = tv.datasets.SVHN(ROOT, download=True, transform=transform)
        source.targets = torch.tensor(source.labels)

        test = tv.datasets.MNIST(ROOT, train=False, download=True,
                                 transform=tv.transforms.Compose([
                                       tv.transforms.Grayscale(3),
                                       transform]))
        target = tv.datasets.MNIST(ROOT, train=True, download=True,
                                   transform=tv.transforms.Compose([
                                       tv.transforms.Grayscale(3),
                                       transform]))
        EPOCHS = 150
        net = svhn_net().to(device)

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

    if args.uda:
        train_loader = unsda_loader
        use_target_labels = False
    else:
        train_loader = mixed_loader
        use_target_labels = True

    return train_loader, test_loader, net, EPOCHS


if __name__ == '__main__':
    # Make the dataset
    train_loader, test_loader, net, EPOCHS = get_setting()

    if args.uda:
        use_target_labels = False
    else:
        use_target_labels = True

    start_epoch = 5
    total_steps = (EPOCHS-start_epoch) * len(train_loader)

    print("Do a validation before starting to check it is ok...")
    val_loss, val_acc, dom_acc = valid(net, valid_loader=test_loader)
    print(f"Epoch {-1:03d} : Test Loss {val_loss:.6f}, Test Acc {val_acc:.2f}, Domain Acc {dom_acc:.2f}")
    print("Result should be random guessing, i.e. 10% accuracy")

    best_val_loss = val_loss
    best_epoch =-1
    best_val_acc = val_acc
    best_model = torch.save(net.state_dict(),  save_name)

    T_d = nn.Parameter(torch.FloatTensor([args.T]).to(device))
    T_c = nn.Parameter(torch.FloatTensor([0]).to(device))

    t_o = optim.SGD([T_d, T_c], lr=0.0)

    # training loop
    for epoch in range(EPOCHS):
        # steps
        start_steps = (epoch-start_epoch) * len(train_loader)

        alpha_d = 0
        if epoch>=start_epoch:
            alpha_d = args.D

        # train epoch
        learning_rate = 0.01 / ((1 + 10 * (epoch) / EPOCHS)**0.75)
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
        # scheduler.step()
        print(f"Learning rate: {learning_rate}")

        if args.revgrad:
            train_loss, train_acc = train_epoch_dann(net, start_steps, total_steps, train_loader=train_loader,
                                                     optimizer=optimizer, use_target_labels=use_target_labels)
        else:
            train_loss, train_acc = train_epoch_snnl(net, start_steps, total_steps, train_loader=train_loader,
                                                     optimizer=optimizer, t_o=t_o, T_d=T_d, T_c=T_c, ALPHA_D=alpha_d,
                                                     use_target_labels=use_target_labels)

        # valid!
        val_loss, val_acc, dom_acc = valid(net, valid_loader=test_loader)
        print(f"Epoch {epoch + 1:03d} : Test Loss {val_loss:.6f}, Test Acc {val_acc:.2f}, Domain Acc {dom_acc:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            best_model = torch.save(net.state_dict(),  save_name)

    with open('results.csv', 'a') as file:
        file.write(f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')},{setting},{method},{EPOCHS},{val_loss},{val_acc},{best_epoch},{best_val_loss},{best_val_acc}\n")


