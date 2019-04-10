import argparse
import config as conf
import os
import torch
import numpy as np
import logging

if not os.path.exists("checkpoint"):
    os.mkdir("checkpoint")
if not os.path.exists("models"):
    os.mkdir("models")
if not os.path.exists("logs"):
    os.mkdir("logs")

# Instantiate the parser to get ARGUMENTS
parser = argparse.ArgumentParser(description='Incremental class learning with Domain Adaptation on OfficeHome dataset.')

# dataset variables
parser.add_argument('setting', default='gtsrb', help='Setting to run (see config.py)')
parser.add_argument('--root', default='/home/fcdl/dataset/', help='Base directory where are stored the data')

parser.add_argument('--to_run', default=2, type=int, help='Number of last run to test (each run has different order')
parser.add_argument('--from_run', default=0, type=int, help='Number of first run to test (each run has different order')

# network/training variables
parser.add_argument('--pretrained', default=None, type=str, help='If start with ImageNet pretraining or not')
parser.add_argument('--epochs', default=None, type=int, help='The number of epochs to use')

# method variables
parser.add_argument('-m', '--method', default='icarl', help='Method to be tested')
parser.add_argument('-d', '--da', default=None, help='Domain adaptation method')
parser.add_argument('-l', '--log', default=None, help='Method name where are saved logs and results')
parser.add_argument('-c', '--config_file', default=None, help='Config file where to get parameters for training')
parser.add_argument('--seed', default=42, type=int, help='The random seed to use')


args = parser.parse_args()

config = conf.get_config(args.setting)

if args.da is not None:
    method_name = args.method + "-" + args.da
else:
    method_name = args.method

if args.log is None:
    if args.config_file is None:
        log = method_name
    else:
        log = args.config_file.split("/")[-1][:-5]
else:
    log = args.log

n_base = config['data_conf']['n_base']
n_incr = config['data_conf']['n_incr']

# fix for reproducibility
torch.manual_seed(args.seed)
np.random.seed(seed=args.seed)

for run in range(args.from_run, args.to_run):
    print(f"Logs will be saved in logs/{args.setting}/run{run}/{log}.train")
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        filename=f"logs/{args.setting}/run{run}/{log}.train", filemode='a')

    # get the data
    data = config['dataset'](args.root, **config['data_conf'], run_number=run, workers=8)
    # define network
    network = conf.get_network(config['network-type'], args.da)(num_classes=config['n_classes'],
                                                                pretrained=args.pretrained)
    # define the method
    method = conf.get_method(args.method, da_method=args.da, config=args.config_file, network=network, n_classes=config['n_classes'],
                             n_base=n_base, n_incr=n_incr, features=config['n_features'],
                             log=f"logs/{args.setting}/run{run}/{log}", name=f"{args.setting}-{run}-{log}")
    # run fit!
    acc = method.fit(data, epochs=args.epochs)
