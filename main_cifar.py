import data
import methods
from networks import networks
from datetime import datetime
import argparse
import os
from torchvision import transforms

if not os.path.exists("checkpoint"):
    os.mkdir("checkpoint")
if not os.path.exists("models"):
    os.mkdir("models")
if not os.path.exists("logs"):
    os.mkdir("logs")

# Instantiate the parser to get ARGUMENTS
parser = argparse.ArgumentParser(description='Incremental class learning.')
# training variables
parser.add_argument('--batch_size', default=128, type=int, help='Bath size for the training')
parser.add_argument('--epochs', default=70, type=int, help='Epochs for the training')
# dataset variables
parser.add_argument('--root', default='/home/fabioc/dataset', help='Base directory where are stored the data')
parser.add_argument('--num_class_batch', default=10, type=int, help='Number of classes each increment')
parser.add_argument('--num_runs', default=10, type=int, help='Number of runs to test (each run has different order')
parser.add_argument('--order', default='data/cifar_order.npy', help='Order file path')
# network variables
parser.add_argument('--depth', default=5, type=int, help='Architecture depth')
# method variables
parser.add_argument('--method', default='icarl', help='Method to be tested')
parser.add_argument('--log', default=None, help='Method name where are saved logs and results')

args = parser.parse_args()

batch_size = args.batch_size  # Batch size
n = args.depth  # Set the depth of the architecture: n = 5 -> 32 layers (See He et al. paper)
nb_cl = args.num_class_batch  # Classes per group
nb_runs = args.num_runs  # Number of runs (random ordering of classes at each run)
DATA_ROOT = args.root
method_name = args.method
if args.log is None:
    log = method_name
else:
    log = args.log

# now start with the main

# create the transforms
# Normalize to have range between -1,1 : (x - 0.5) * 2
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
# Create data augmentation transform
augmentation = transforms.Compose([transforms.RandomHorizontalFlip(),
                                   transforms.RandomCrop((32, 32), padding=4)])

for run in range(nb_runs):
    # get the data
    data = data.ICIFAR(args.root,
                       num_cl_first=nb_cl, num_cl_after=nb_cl,
                       augmentation=augmentation, transform=transform,
                       batch_size=batch_size, run_number=run, workers=8)
    # define network
    network = networks.CifarResNet()
    # define the method
    method = methods.get_method(method_name)(network)
    # run fit!
    acc = method.fit(data, nb_cl, epochs=args.epochs, log_name=f"log/{log}_run")

