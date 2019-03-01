import data
import methods
from networks import networks

import os
if not os.path.exists("checkpoint"):
    os.mkdir("checkpoint")

# Modifiable Settings ##########
# todo argparse
batch_size = 128  # Batch size
n = 5  # Set the depth of the architecture: n = 5 -> 32 layers (See He et al. paper)
nb_cl = 10  # Classes per group
nb_runs = 10  # Number of runs (random ordering of classes at each run)
DATA_ROOT = "/home/fcdl/dataset"
#

# get the data
icifar = data.get_dataset('ICIFAR')(DATA_ROOT, batch_size, nb_cl, 'data/fixed_order.npy')
accuracies = []

for run in range(nb_runs):
    # define the run
    icifar.set_run(run)
    # define network
    network = networks.CifarResNet()
    # define the method
    icarl = methods.get_method('ICarl')(network)
    # run fit!
    acc = icarl.fit(icifar, nb_cl)
    accuracies.append(acc)

print(accuracies)
