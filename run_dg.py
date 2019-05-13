from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torchvision as tv
from torchvision import transforms
import datetime

from data import multi
from data import DoubleDataset
from data.common import get_index_of_classes
from networks.networks import resnet18, resnet50
from train import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--suffix', default="0", help='The suffix for the name of experiment')
parser.add_argument('-D', default=1, type=float)
parser.add_argument('-Y', default=0, type=float)
parser.add_argument('-T', default=0, type=float)
parser.add_argument('--revgrad', action='store_true')
parser.add_argument('--dataset', default="office")
parser.add_argument('--sources', default="pac")
parser.add_argument('--target', default="r")
args = parser.parse_args()

# parameters and utils
device = 'cuda'
ROOT = '/home/fcdl/dataset/'
setting = f"dg-{args.dataset}"
method = 'dann' if args.revgrad else f'snnl-{args.D}-{args.T}'
method += f"_{args.suffix}"
save_name = f"models/{setting}/method.pth"

if __name__ == '__main__':
    # Make the dataset
    if args.dataset == 'office':
        source, target = multi.office_home(ROOT, args.sources, args.target)
        EPOCHS = 60
        num_classes = 65
        net = resnet50(pretrained=True, num_classes=num_classes).to(device)

    elif args.dataset == 'pacs':
        source, target = multi.pacs(ROOT, args.sources, args.target)
        EPOCHS = 150
        num_classes = 7
        net = resnet18(pretrained=True, num_classes=num_classes).to(device)
    else:
        assert False, "Dataset not found"

    train_loader = DataLoader(source, 64, True, drop_last=True)
    test_loader = DataLoader(target, 64, False, drop_last=False)

    start_epoch = 5
    total_steps = (EPOCHS-start_epoch) * len(train_loader)

    print("Do a validation before starting to check it is ok...")
    val_loss, val_acc, dom_acc = valid(net, valid_loader=test_loader)
    print(f"Epoch {-1:03d} : Test Loss {val_loss:.6f}, Test Acc {val_acc:.2f}, Domain Acc {dom_acc:.2f}")
    print(f"Result should be random guessing, i.e. {100./num_classes:.2f}% accuracy")

    best_val_loss = val_loss
    best_epoch = -1
    best_val_acc = val_acc
    best_model = torch.save(net.state_dict(),  save_name)

    T_d = nn.Parameter(torch.FloatTensor([args.T]).to(device))
    T_c = nn.Parameter(torch.FloatTensor([0]).to(device))

    t_o = optim.SGD([T_d, T_c], lr=0.0)

    # training loop
    for epoch in range(EPOCHS):
        # steps
        start_steps = (epoch-start_epoch) * len(train_loader)

        alpha_d = args.D

        # train epoch
        learning_rate = 0.01 / ((1 + 10 * (epoch) / EPOCHS)**0.75)
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
        # scheduler.step()
        print(f"Learning rate: {learning_rate}")

        if args.revgrad:
            train_loss, train_acc = train_epoch_dann_dg(net, start_steps, total_steps, train_loader=train_loader,
                                                        optimizer=optimizer)
        else:
            train_loss, train_acc = train_epoch_snnl_dg(net, start_steps, total_steps, train_loader=train_loader,
                                                        optimizer=optimizer, t_o=t_o, T_d=T_d, T_c=T_c, ALPHA_D=alpha_d,
                                                        ALPHA_Y=args.Y)

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


