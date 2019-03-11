import data
import methods
from networks import networks
from datetime import datetime
import argparse
import os

if not os.path.exists("checkpoint"):
    os.mkdir("checkpoint")
if not os.path.exists("models"):
    os.mkdir("models")

# Instantiate the parser to get ARGUMENTS
parser = argparse.ArgumentParser(description='Incremental class learning.')
# training variables
parser.add_argument('--batch_size', default=128, type=int, help='Bath size for the training')
parser.add_argument('--epochs', default=70, type=int, help='Epochs for the training')
# dataset variables
parser.add_argument('--dataset', default='icifar', help='Dataset name')
parser.add_argument('--root', default='/home/fabioc/dataset', help='Base directory where are stored the data')
parser.add_argument('--num_class_batch', default=10, type=int, help='Number of classes each increment')
parser.add_argument('--num_runs', default=10, type=int, help='Number of runs to test (each run has different order')
parser.add_argument('--order', default=None, help='Order file path')  # 'data/cifar_order.npy'
# network variables
parser.add_argument('--depth', default=5, type=int, help='Architecture depth')
# method variables
parser.add_argument('--method', default='icarl', help='Method to be tested')

args = parser.parse_args()

batch_size = args.batch_size  # Batch size
n = args.depth  # Set the depth of the architecture: n = 5 -> 32 layers (See He et al. paper)
nb_cl = args.num_class_batch  # Classes per group
nb_runs = args.num_runs  # Number of runs (random ordering of classes at each run)
DATA_ROOT = args.root
method_name = args.method
dataset_name = args.dataset

# now start with the main

# get the data
dataset = data.get_dataset(dataset_name)(DATA_ROOT, batch_size, nb_cl, args.order)
accuracies = []

for run in range(nb_runs):
    # define the run
    dataset.set_run(run)
    # define network
    network = networks.CifarResNet()
    # define the method
    method = methods.get_method(method_name)(network)
    # run fit!
    acc = method.fit(dataset, nb_cl, epochs=args.epochs)
    accuracies.append(acc)

with open(f"{method_name}.txt", "a") as f:
    f.write(str(datetime.utcnow()))
    f.write(str(accuracies))
