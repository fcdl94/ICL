from networks.networks import *
from data import *
from torchvision import transforms
from methods.icarl import ICarl
import json
from methods.fine_tuning import FineTuning
from methods.icarl_da import ICarlDA


gtsrb_train = 'GTSRB/Final_Training/Images'
gtsrb_test = 'GTSRB/Final_Test'
synt_sign = 'synthetic_data'

sk_ph_train = 'sketchy/photo_train'
sk_ph_test = 'sketchy/photo_test'
sk_ph_full = 'sketchy/photo'
sk_sk_train = 'sketchy/sketch_train'
sk_sk_test = 'sketchy/sketch_test'
sk_sk_full = 'sketchy/sketch'

config = {
    ############     CIFAR     ###################
    'icifar': {
        'n_classes': 100,
        'n_features': 64,
        'data_conf': {
            'target': None,
            'test': None,
            'source': None,
            'validation_size': 0.2,
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
            'validation_size': 0.2,
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
            'n_base': 0,  # 0 base
            'n_incr': 43,  # 1 incremental
            'validation_size': 0.2
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
            'validation_size': 0.2
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
            'validation_size': 0.2
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
            'validation_size': 0.2
        },
        'network-type': "cifar",
        'dataset': IDADataloader
    },  # ICL+DA setting (13 - 3*10)

    #TODO MAKE SPLIT in DATA! ############ Sketchy ####################
    'sketchy-ph': {
        'n_classes': 125,
        'n_features': 256,
        'data_conf': {
            'target': sk_ph_train,
            'test': sk_ph_test,
            'n_base': 125,  # 1 base
            'n_incr': 0,    # 0 incremental
            'validation_size': 0.2
        },
        'network-type': "wide",
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
            'validation_size': 0.2
        },
        'network-type': "wide",
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
            'validation_size': 0.2
        },
        'network-type': "wide",
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
            'validation_size': 0.2
        },
        'network-type': "wide",
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
            'validation_size': 0.2
        },
        'network-type': "wide",
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
        },
        'network-type': "wide",
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
            'validation_size': 0.2
        },
        'network-type': "wide",
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
        },
        'network-type': "wide",
        'dataset': IDADataloader
    }  # ICL+DA Photo -> Sketch (50 - 3*25)
}


def __parse_config__(config):
    pars = {}

    if config is not None:
        print("Using config file: " + config)
        with open(config, "r") as read_file:
            pars = json.load(read_file)
    return pars


def get_method(m_name, config=None, **kwargs):

    pars = __parse_config__(config)

    for i in kwargs:
        pars[i] = kwargs[i]

    if m_name.lower() == 'icarl-single':
        return ICarl(**pars)
    if m_name.lower() == 'icarl':
        return ICarlDA(**pars)
    if m_name.lower() == 'lwf':
        return ICarl(**pars, mem_size=0)
    if m_name.lower() == 'finetuning':
        return FineTuning(**pars)

    assert True, f"There is no methods called {m_name}."


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
    elif 'sketchy' in name or 'dense' in name:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        # Normalize to have range between -1,1 : (x - 0.5) * 2
        transform = transforms.Compose([transforms.Resize((64, 64)),
                                        transforms.ToTensor(),
                                        normalize])
        # Create data augmentation transform
        augmentation = transforms.Compose([transforms.RandomResizedCrop((64, 64), (0.6, 1.)),
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

    return transform, augmentation


def get_network(typ, da):
    if "cifar" in typ:
        if da is None:
            return cifar_resnet
        elif "dial" in da:
            return cifar_resnet_dial

    elif "wide" in typ:
        if da is None:
            return wide_resnet_dial
        elif "dial" in da:
            return wide_resnet
        else:
            return wide_resnet_dial
    else:
        raise NotImplementedError


def get_config(name):
    assert name in config, "Configuration name not found."
    config[name]['data_conf']['transform'], config[name]['data_conf']['augmentation'] = get_transform(name)
    return config[name]


old_config = {
    # Office
    'office-rw': {
        'n_classes': 65,
        'n_features': 512,
        'data_conf': {
            'target': 'office/Real World',
            'n_base': 65,  # 1 base
            'n_incr': 0,  # 0 incremental
            'order_file': 'office_order.npy'
        },
        'network': resnet18,
        'dataset': IncrementalDataloader
    },
    'ioffice-rw-pr': {
        'n_classes': 65,
        'n_features': 512,
        'data_conf': {
            'target': 'office/Real World',
            'source': 'office/Product',
            'n_base': 20,  # 1 base
            'n_incr': 15,  # 3 incremental
            'order_file': 'office/office_order.npy'
        },
        'network': resnet18,
        'dataset': IDADataloader
    },
    'syns': {
        'n_classes': 43,
        'n_features': 64,
        'data_conf': {
            'target': synt_sign,
            'n_base': 43,  # 1 base
            'n_incr': 0,  # 0 incremental
            'validation_size': 0.2
        },
        'network-type': "cifar",
        'dataset': IncrementalDataloader
    },
    'isyns': {
        'n_classes': 43,
        'n_features': 64,
        'data_conf': {
            'target': synt_sign,
            'n_base': 13,  # 1 base
            'n_incr': 10,  # 3 incremental
            'validation_size': 0.2
        },
        'network-type': "cifar",
        'dataset': IncrementalDataloader
    },
}
