from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torchvision as tv
from torchvision import transforms
from torchvision.datasets import ImageFolder
import datetime
import torch
from data import MNISTM
from data import DoubleDataset
from data.common import get_index_of_classes
from networks.svhn import lenet_net, svhn_net
from train import *
from logger import TensorboardXLogger as Log
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--suffix', default="0", help='The suffix for the name of experiment')
parser.add_argument('-D', default=1, type=float)
parser.add_argument('-Y', default=0, type=float)
parser.add_argument('-T', default=0, type=float)
parser.add_argument('--revgrad', action='store_true')
parser.add_argument('--dataset', default="mnist")
parser.add_argument('--uda', action='store_true')
parser.add_argument('--so', action='store_true')
parser.add_argument('-s', '--source', default="p")
parser.add_argument('-t', '--target', default="r")
parser.add_argument('--start_epoch', default=0, type=int)

args = parser.parse_args()

assert not (args.revgrad and args.so), "Please, use only one between Revgrad and SO"

# parameters and utils
device = 'cuda'
ROOT = '/home/fcdl/dataset/'
setting = f"{'uda' if args.uda else 'mixed'}-{args.dataset}/{args.source}-{args.target}"
method = 'dann' if args.revgrad else f'snnl-d{args.D:.1f}-t{args.T:.1f}'
method += f"_{args.suffix}"
save_name = f"models/{setting}/{method}.pth"

os.makedirs(f"models/{setting}/", exist_ok=True)

n_classes = 0

def get_setting():
    global n_classes
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
        batch_size = 128
        n_classes = 10
    elif args.dataset == 'office':
        from networks.networks import resnet50
        paths = {"p": ROOT + "office/Product",
                 "a": ROOT + "office/Art",
                 "c": ROOT + "office/Clipart",
                 "r": ROOT + "office/Real World"}

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # Normalize to have range between -1,1 : (x - 0.5) * 2
        transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        normalize])
        # Create data augmentation transform
        augmentation = transforms.Compose([transforms.Resize(256),
                                           transforms.RandomResizedCrop(224, (0.6, 1.)),
                                           transforms.RandomHorizontalFlip(),
                                           transform])

        source = ImageFolder(paths[args.source], augmentation)
        target = ImageFolder(paths[args.target], augmentation)

        test = ImageFolder(paths[args.target], transform) 
        EPOCHS = 60
        n_classes = 65
        net = resnet50(pretrained=True, num_classes=65).to(device)
        batch_size = 32
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
        batch_size = 128
        n_classes = 10
        net = svhn_net().to(device)

    # target_loader = DataLoader(target, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=8)
    # source_loader = DataLoader(source, batch_size=batch_size, shuffle=True, num_workers=8)

    # make the hybrid dataset here.
    unsda = DoubleDataset(source, target)
    unsda_loader = DataLoader(unsda, batch_size//2, True, num_workers=8)

    # get mnist [0:4], mnistm[4:9]
    indices = get_index_of_classes(torch.tensor(target.targets), list(range(0, 5)))
    half_target = Subset(target, indices)

    indices = get_index_of_classes(torch.tensor(source.targets), list(range(5, 10)))
    half_source = Subset(source, indices)

    mixed = DoubleDataset(half_source, half_target)
    mixed_loader = DataLoader(mixed, batch_size//2, True, num_workers=8, drop_last=True)

    if args.uda:
        train_loader = unsda_loader
        use_target_labels = False
    else:
        train_loader = mixed_loader
        use_target_labels = True

    return train_loader, test_loader, net, EPOCHS


if __name__ == '__main__':
    # create the Logger
    log = Log(f'logs/{setting}', method)

    # Make the dataset
    train_loader, test_loader, net, EPOCHS = get_setting()

    if args.uda:
        use_target_labels = False
    else:
        use_target_labels = True

    start_epoch = args.start_epoch
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
        if epoch >= start_epoch:
            alpha_d = args.D

        # train epoch
        learning_rate = 0.001 / ((1 + 10 * (epoch) / EPOCHS)**0.75)
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
        # scheduler.step()
        print(f"Learning rate: {learning_rate}")

        if args.revgrad:
            train_loss, train_acc = train_epoch_dann(net, start_steps, total_steps, train_loader=train_loader,
                                                     optimizer=optimizer, use_target_labels=use_target_labels)
            dom_loss, class_loss = 0., 0.
        elif args.so:
            train_loss, train_acc = train_epoch_single(net, start_steps, total_steps, train_loader=train_loader,
                                                     optimizer=optimizer, use_target_labels=use_target_labels)
            dom_loss, class_loss = 0., 0.
        else:
            train_loss, train_acc, dom_loss, class_loss = train_epoch_snnl(net, start_steps, total_steps, train_loader=train_loader,
                                                     optimizer=optimizer, t_o=t_o, T_d=T_d, T_c=T_c, ALPHA_D=alpha_d,
                                                     use_target_labels=use_target_labels)


        # valid!
        val_loss, val_acc, dom_acc = valid(net, valid_loader=test_loader)
        print(f"Epoch {epoch + 1:03d} : Test Loss {val_loss:.6f}, Test Acc {val_acc:.2f}, Domain Acc {dom_acc:.2f}")

        log.log_training(epoch, train_loss, train_acc, val_loss, val_acc, dom_loss, class_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            best_model = torch.save(net.state_dict(),  save_name)

    val_loss, val_acc, dom_acc = valid(net, valid_loader=test_loader, conf_matrix=True, log=log, n_classes=n_classes)
    with open('results.csv', 'a') as file:
        file.write(f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')},{setting},{method},{EPOCHS},{val_loss},{val_acc},{best_epoch},{best_val_loss},{best_val_acc}\n")


