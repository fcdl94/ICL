from data import IncrementalDataloader as IDataloader
import methods
from networks import networks
import argparse
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch
import numpy as np

if not os.path.exists("checkpoint"):
    os.mkdir("checkpoint")
if not os.path.exists("models"):
    os.mkdir("models")
if not os.path.exists("logs"):
    os.mkdir("logs")

# Instantiate the parser to get ARGUMENTS
parser = argparse.ArgumentParser(description='Incremental class learning with Domain Adaptation on OfficeHome dataset.')

# dataset variables
parser.add_argument('--root', default='/home/fcdl/dataset/office', help='Base directory where are stored the data')
parser.add_argument('-t', '--target', default='Real World', help='Dataset Folder')
parser.add_argument('-b', '--num_base_classes', default=10, type=int, help='Number of classes each increment')
parser.add_argument('-i', '--num_incremental_classes', default=5, type=int, help='Number of classes each increment')
parser.add_argument('--num_runs', default=1, type=int, help='Number of runs to test (each run has different order')
parser.add_argument('--from_run', default=0, help='The first run (order of classes) to be evaluated')
parser.add_argument('--order', default=None, help='Order file path')

parser.add_argument('--batch_size', default=16, type=int, help='Bath size for the training')

# network variables
parser.add_argument('--pretrained', default=False, type=bool, help='If start with ImageNet pretraining or not')

# method variables
parser.add_argument('-m', '--method', default='icarl', help='Method to be tested')
parser.add_argument('-l', '--log', default=None, help='Method name where are saved logs and results')
parser.add_argument('-c', '--config_file', default=None, help='Config file where to get parameters for training')
parser.add_argument('--seed', default=42, help='The random seed to use')
parser.add_argument('--epochs', default=None, help='The number of epochs to use')

args = parser.parse_args()

batch_size = args.batch_size  # Batch size

nb_base = args.num_base_classes  # Base classes per group (first iteration)
nb_incr = args.num_incremental_classes

nb_runs = args.num_runs  # Number of runs (random ordering of classes at each run)
DATA_ROOT = args.root
method_name = args.method
if args.log is None:
    log = method_name
else:
    log = args.log

# now start with the main

# fix for reproducibility
torch.manual_seed(args.seed)
np.random.seed(seed=args.seed)

# create the transforms
# normalization of ImageNet
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Normalize to have range between -1,1 : (x - 0.5) * 2
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                normalize])
# Create data augmentation transform
augmentation = transforms.Compose([transforms.Resize(250),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomCrop((224, 224))])

for run in range(int(args.from_run), nb_runs):
    # get the data
    target = ImageFolder(args.root + "/" + args.target, None, None)

    data = IDataloader(target,
                       num_cl_first=nb_base, num_cl_after=nb_incr,
                       augmentation=augmentation, transform=transform,
                       batch_size=batch_size, run_number=run, workers=8)
    # define network
    network = networks.resnet18(pretrained=args.pretrained, num_classes=65)
    # define the method
    method = methods.get_method(method_name, config=args.config_file, network=network, n_classes=65,
                                nb_base=nb_base, nb_incr=nb_incr, features=512,
                                log=f"logs/reals/{log}/run{run}")
    # run fit!
    acc = method.fit(data, epochs=args.epochs)
