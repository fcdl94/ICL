from networks.networks import *
from networks.svhn import *
from networks.gtsrb import *
from data import *
from torchvision import transforms
from methods.icarl import ICarl
import json
from methods.fine_tuning import FineTuning
from methods.icarl_da import ICarlDA
from methods.icarl_revgrad import ICarlRG


gtsrb_train = 'GTSRB/Final_Training/Images'
gtsrb_test = 'GTSRB/Final_Test'
synt_sign = 'synthetic_data'
gtsrb_order = 'GTSRB/gtsrb_order.csv'

sk_ph_train = 'sketchy/photo_train'
sk_ph_test = 'sketchy/photo_test'
sk_ph_full = 'sketchy/photo'
sk_sk_train = 'sketchy/sketch_train'
sk_sk_test = 'sketchy/sketch_test'
sk_sk_full = 'sketchy/sketch'
sk_order = 'sketchy/sketchy_order.csv'

digits_order = 'digits_order.csv'

sketchy_net_type = 'resnet50'

validation_size = 0.2

config = {
    ############     CIFAR     ###################
    'icifar': {
        'n_classes': 100,
        'n_features': 64,
        'data_conf': {
            'target': None,
            'test': None,
            'source': None,
            'validation_size': validation_size,
            'n_base': 10,   # 1 base
            'n_incr': 10,   # 9 incremental
        },
        'network-type': "cifar",
        'dataset': CifarDataloader
    },  # Incremental CIFAR - no DA (10 - 9*10)
    'cifar': {
        'n_classes': 100,
        'n_features': 64,
        'data_conf': {
            'target': None,
            'test': None,
            'source': None,
            'validation_size': validation_size,
            'n_base': 100,   # 1 base
            'n_incr': 0,     # 0 incremental
        },
        'network-type': "cifar",
        'dataset': CifarDataloader
    },   # traditional CIFAR - no DA, no ICL

    ############ Traffic signs ###################
    'gtsrb': {
        'n_classes': 43,
        'n_features': 64,
        'data_conf': {
            'target': gtsrb_train,
            'test': gtsrb_test,
            'source': None,
            'n_base': 43,  # 0 base
            'n_incr': 0,  # 1 incremental
            'validation_size': validation_size,
            'order_file': gtsrb_order,
        },
        'network-type': "cifar",
        'dataset': SingleDataloader
    },           # traditional setting, no DA, no ICL
    'igtsrb': {
        'n_classes': 43,
        'n_features': 64,
        'data_conf': {
            'target': gtsrb_train,
            'test': gtsrb_test,
            'source': None,
            'n_base': 13,  # 0 base
            'n_incr': 10,  # 1 incremental
            'validation_size': validation_size,
            'order_file': gtsrb_order,
        },
        'network-type': "cifar",
        'dataset': SingleDataloader
    },          # ICL setting (13 - 3*10)
    'syns-to-gtsrb': {  # dummy setting, just to check if there's Domain shift
        'n_classes': 43,
        'n_features': 64,
        'data_conf': {
            'target': synt_sign,
            'test': gtsrb_test,
            'source': None,
            'n_base': 43,  # 0 base
            'n_incr': 0,  # 1 incremental
            'validation_size': validation_size,
            'order_file': gtsrb_order,
        },
        'network-type': "cifar",
        'dataset': SingleDataloader
    },   # full DA setting
    'isyns-to-gtsrb': {
        'n_classes': 43,
        'n_features': 64,
        'data_conf': {
            'target': gtsrb_train,
            'test': gtsrb_test,
            'source': synt_sign,
            'n_base': 13,  # 1 base
            'n_incr': 10,  # 3 incremental
            'validation_size': validation_size,
            'order_file': gtsrb_order,
        },
        'network-type': "cifar",
        'dataset': IDADataloader
    },  # ICL+DA setting (13 - 3*10)

    ############# Sketchy ####################
    'sketchy-ph': {
        'n_classes': 125,
        'n_features': 256,
        'data_conf': {
            'target': sk_ph_train,
            'test': sk_ph_test,
            'n_base': 125,  # 1 base
            'n_incr': 0,    # 0 incremental
            'validation_size': validation_size,
            'order_file': sk_order,
            'batch_size': 32
        },
        'network-type': sketchy_net_type,
        'dataset': SingleDataloader
    },  # traditional setting, no DA, no ICL, on Photo
    'isketchy-ph': {
        'n_classes': 125,
        'n_features': 256,
        'data_conf': {
            'target': sk_ph_train,
            'test': sk_ph_test,
            'n_base': 50,  # 1 base
            'n_incr': 25,  # 0 incremental
            'validation_size': validation_size,
            'order_file': sk_order,
            'batch_size': 32
        },
        'network-type': sketchy_net_type,
        'dataset': SingleDataloader
    },  # ICL setting on photo (50 - 3*25)
    'sketchy-sk': {
        'n_classes': 125,
        'n_features': 256,
        'data_conf': {
            'target': sk_sk_train,
            'test': sk_sk_test,
            'n_base': 125,  # 1 base
            'n_incr': 0,  # 0 incremental
            'validation_size': validation_size,
            'order_file': sk_order,
            'batch_size': 32
        },
        'network-type': sketchy_net_type,
        'dataset': SingleDataloader
    },  # traditional setting, no DA, no ICL, on Sketch
    'isketchy-sk': {
        'n_classes': 125,
        'n_features': 256,
        'data_conf': {
            'target': sk_sk_train,
            'test': sk_sk_test,
            'n_base': 50,  # 1 base
            'n_incr': 25,  # 3 incremental
            'validation_size': validation_size,
            'order_file': sk_order,
            'batch_size': 32
        },
        'network-type': sketchy_net_type,
        'dataset': SingleDataloader
    },  # ICL setting on Sketch (50 - 3*25)
    'sketchy-sk-to-ph': {
        'n_classes': 125,
        'n_features': 256,
        'data_conf': {
            'target': sk_sk_full,
            'test': sk_ph_test,
            'n_base': 125,  # 1 base
            'n_incr': 0,    # 0 incremental
            'validation_size': validation_size,
            'order_file': sk_order,
            'batch_size': 32
        },
        'network-type': sketchy_net_type,
        'dataset': SingleDataloader
    },  # full DA setting Sketch -> Photo
    'isketchy-sk-to-ph': {
        'n_classes': 125,
        'n_features': 256,
        'data_conf': {
            'target': sk_ph_train,
            'source': sk_sk_full,
            'test': sk_ph_test,
            'n_base': 50,  # 1 base
            'n_incr': 25,  # 0 incremental
            'validation_size': validation_size,
            'order_file': sk_order,
            'batch_size': 32
        },
        'network-type': sketchy_net_type,
        'dataset': IDADataloader
    },  # ICL+DA Sketch -> Photo (50 - 3*25)
    'sketchy-ph-to-sk': {
        'n_classes': 125,
        'n_features': 256,
        'data_conf': {
            'target': sk_ph_full,
            'test': sk_sk_test,
            'n_base': 125,  # 1 base
            'n_incr': 0,  # 0 incremental
            'validation_size': validation_size,
            'order_file': sk_order,
            'batch_size': 32
        },
        'network-type': sketchy_net_type,
        'dataset': SingleDataloader
    },   # full DA setting Photo -> Sketch
    'isketchy-ph-to-sk': {
        'n_classes': 125,
        'n_features': 256,
        'data_conf': {
            'target': sk_sk_train,
            'source': sk_ph_full,
            'test': sk_sk_test,
            'n_base': 50,  # 1 base
            'n_incr': 25,  # 0 incremental
            'validation_size': validation_size,
            'order_file': sk_order,
            'batch_size': 32
        },
        'network-type': sketchy_net_type,
        'dataset': IDADataloader
    },  # ICL+DA Photo -> Sketch (50 - 3*25)

    ## DIGITS ##
    "isvhn-mnist": {
        'n_classes': 10,
        'n_features': 128*3*3,
        'data_conf': {
            'target': None,
            'test': None,
            'n_base': 5,  # 1 base
            'n_incr': 5,  # 0 incremental
            'validation_size': validation_size,
            'order_file': digits_order,
        },
        'network-type': "svhn",
        'dataset': MNISTDataloader
    },
    "isvhn": {
        'n_classes': 10,
        'n_features': 128*3*3,
        'data_conf': {
            'target': None,
            'test': None,
            'n_base': 5,  # 1 base
            'n_incr': 5,  # 0 incremental
            'validation_size': validation_size,
            'order_file': digits_order,
        },
        'network-type': "svhn",
        'dataset': SVHN_to_MNIST_Dataloader
    },
    "imnistm-mnist": {
        'n_classes': 10,
        'n_features': 48 * 4 * 4,
        'data_conf': {
            'target': None,
            'test': None,
            'n_base': 5,  # 1 base
            'n_incr': 5,  # 0 incremental
            'validation_size': validation_size,
            'order_file': digits_order,
        },
        'network-type': "mnist",
        'dataset': MNISTDataloader
    },
    "imnistm": {
        'n_classes': 10,
        'n_features': 48 * 4 * 4,
        'data_conf': {
            'target': None,
            'test': None,
            'n_base': 5,  # 1 base
            'n_incr': 5,  # 0 incremental
            'validation_size': validation_size,
            'order_file': digits_order,
        },
        'network-type': "mnist",
        'dataset': MNISTM_to_MNIST_Dataloader
    }
}


def __parse_config__(config):
    pars = {}

    if config is not None:
        print("Using config file: " + config)
        with open(config, "r") as read_file:
            pars = json.load(read_file)
    return pars


def get_method(m_name, da_method=None, config=None, **kwargs):

    pars = __parse_config__(config)

    for i in kwargs:
        pars[i] = kwargs[i]

    if m_name.lower() == 'icarl-single':
        return ICarl(**pars)
    if m_name.lower() == 'icarl' and (da_method is None or "dial" in da_method):
        return ICarlDA(**pars)
    if m_name.lower() == 'icarl' and "revgrad" in da_method:
        return ICarlRG(**pars)
    if m_name.lower() == 'lwf' and (da_method is None or "dial" in da_method):
        return ICarlDA(**pars, protos=False)
    if m_name.lower() == 'lwf' and "revgrad" in da_method:
        return ICarlRG(**pars, protos=False)
    if m_name.lower() == 'finetuning':
        return FineTuning(**pars)

    assert False, f"There is no methods called {m_name} that uses {da_method} as domain adaptation."


def get_transform(name):
    transform, augmentation = None, None
    if name in ['cifar', 'icifar']:
        # Normalize to have range between -1,1 : (x - 0.5) * 2
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        # Create data augmentation transform
        augmentation = transforms.Compose([transforms.RandomHorizontalFlip(),
                                           transforms.RandomCrop((32, 32), padding=4)])
    elif 'gtsrb' in name or "syns" in name:
        # Normalize to have range between -1,1 : (x - 0.5) * 2
        transform = transforms.Compose([transforms.Resize((32, 32)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        # Create data augmentation transform
        augmentation = transforms.Compose([transforms.Resize((35, 35)),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomCrop((32, 32))])
    elif 'sketchy' in name:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        # Normalize to have range between -1,1 : (x - 0.5) * 2
        transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        normalize])
        # Create data augmentation transform
        augmentation = transforms.Compose([transforms.RandomResizedCrop(224, (0.6, 1.)),
                                           transforms.RandomHorizontalFlip()])
    elif 'office' in name:
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
    elif 'svhn' in name or 'mnist' in name:
        transform = transforms.Compose([transforms.Resize((28, 28)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        augmentation = transforms.Compose([transforms.RandomCrop(28)])

    return transform, augmentation


def get_network(typ, da):
    if "cifar" in typ:
        if da is None:
            return cifar_resnet
        elif "dial" in da:
            return cifar_resnet_dial
        elif "revgrad" in da:
            return cifar_resnet_revgrad
        else:
            return NotImplementedError
    elif "wide" in typ:
        if da is None:
            return wide_resnet_dial
        elif "dial" in da:
            return wide_resnet
        elif "revgrad" in da:
            return wide_resnet_revgrad
        else:
            return NotImplementedError
    elif "svhn" in typ:
        if da is None or "revgrad" in da:
            return svhn_net
        elif "dial" in da:
            return svhn_net_dial
        else:  # "dial" in da:
            return NotImplementedError
    elif "mnist" in typ:
        if da is None or "revgrad" in da:
            return lenet_net
        else:
            return NotImplementedError
    elif "gtsrb" in typ:
        if da is None or "revgrad" in da:
            return gtsrb_net
        elif "dial" in da:
            return gtsrb_net_dial
        else:  #
            return NotImplementedError
    elif "resnet50" in typ:
        if da is None or "revgrad" in da:
            return resnet50
        elif "dial" in da:
            return resnet50_dial
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def get_config(name):
    assert name in config, "Configuration name not found."
    config[name]['data_conf']['transform'], config[name]['data_conf']['augmentation'] = get_transform(name)
    return config[name]
