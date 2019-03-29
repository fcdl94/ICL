from networks.networks import *
from data import *
from torchvision import transforms

config = {
    # CIFAR
    'icifar': {
        'n_classes': 100,
        'n_features': 64,
        'data_conf': {
            'n_base': 10,   # 1 base
            'n_incr': 10,   # 9 incremental
        },
        'network': cifar_resnet,
        'dataset': ICIFAR
    },
    'cifar': {
        'n_classes': 100,
        'n_features': 64,
        'data_conf': {
            'n_base': 100,   # 1 base
            'n_incr': 0,   # 0 incremental
        },
        'network': cifar_resnet,
        'dataset': ICIFAR
    },
    # Traffic signs
    'gtsrb': {
        'n_classes': 43,
        'n_features': 64,
        'data_conf': {
            'target': 'GTSRB/Final Training/Images',
            'n_base': 43,   # 1 base
            'n_incr': 0,   # 0 incremental
            'validation_size': 0.2
        },
        'network': cifar_resnet,
        'dataset': IncrementalDataloader
    },
    'igtsrb': {
        'n_classes': 43,
        'n_features': 64,
        'data_conf': {
            'target': 'GTSRB/Final Training/Images',
            'n_base': 13,   # 1 base
            'n_incr': 10,   # 3 incremental
        },
        'network': cifar_resnet,
        'dataset': IncrementalDataloader
    },
    'syns': {
        'n_classes': 43,
        'n_features': 64,
        'data_conf': {
            'target': 'synthetic_data',
            'n_base': 43,   # 1 base
            'n_incr': 0,   # 0 incremental
            'validation_size': 0.2
        },
        'network': cifar_resnet,
        'dataset': IncrementalDataloader
    },
    'isyns': {
        'n_classes': 43,
        'n_features': 64,
        'data_conf': {
            'target': 'synthetic_data',
            'n_base': 13,  # 1 base
            'n_incr': 10,  # 3 incremental
            'validation_size': 0.2
        },
        'network': cifar_resnet,
        'dataset': IncrementalDataloader
    },
    'syns-to-gtsrb': {
        'n_classes': 43,
        'n_features': 64,
        'data_conf': {
            'target': 'GTSRB/Final Training/Images',
            'source': 'synthetic_data',
            'n_base': 0,  # 0 base
            'n_incr': 43,  # 1 incremental
            'validation_size': 0.2
        },
        'network': cifar_resnet,
        'dataset': IDADataloader
    },
    'isyns-to-gtsrb': {
        'n_classes': 43,
        'n_features': 64,
        'data_conf': {
            'target': 'GTSRB/Final Training/Images',
            'source': 'synthetic_data',
            'n_base': 13,  # 1 base
            'n_incr': 10,  # 3 incremental
            'validation_size': 0.2
        },
        'network': cifar_resnet,
        'dataset': IDADataloader
    },
    # Office
    'office-rw': {
        'n_classes': 65,
        'n_features': 512,
        'data_conf': {
            'target':  'office/Real World',
            'n_base': 65,   # 1 base
            'n_incr': 0,   # 0 incremental
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
    # SKETCHY
    'sketchy-ph': {
        'n_classes': 125,
        'n_features': 256,
        'data_conf': {
            'target': 'sketchy/photo',
            'n_base': 125,  # 1 base
            'n_incr': 0,    # 0 incremental
        },
        'network': wide_resnet,
        'dataset': IncrementalDataloader
    },
    'isketchy-ph': {
        'n_classes': 125,
        'n_features': 256,
        'data_conf': {
            'target': 'sketchy/photo',
            'n_base': 50,  # 1 base
            'n_incr': 25,    # 0 incremental
        },
        'network': wide_resnet,
        'dataset': IncrementalDataloader
    },
    'sketchy-sk': {
        'n_classes': 125,
        'n_features': 256,
        'data_conf': {
            'target': 'sketchy/sketch',
            'n_base': 125,  # 1 base
            'n_incr': 0,    # 0 incremental
        },
        'network': wide_resnet,
        'dataset': IncrementalDataloader
    },
    'isketchy-sk': {
        'n_classes': 125,
        'n_features': 256,
        'data_conf': {
            'target': 'sketchy/sketch',
            'n_base': 50,  # 1 base
            'n_incr': 25,  # 0 incremental
        },
        'network': wide_resnet,
        'dataset': IncrementalDataloader
    },
    'sketchy-sk-to-ph': {
        'n_classes': 125,
        'n_features': 256,
        'data_conf': {
            'target': 'sketchy/photo',
            'source': 'sketchy/sketch',
            'n_base': 0,  # 1 base
            'n_incr': 125,    # 0 incremental
        },
        'network': wide_resnet,
        'dataset': IDADataloader
    },
    'isketchy-sk-to-ph': {
        'n_classes': 125,
        'n_features': 256,
        'data_conf': {
            'target': 'sketchy/photo',
            'source': 'sketchy/sketch',
            'n_base': 50,  # 1 base
            'n_incr': 25,  # 0 incremental
        },
        'network': wide_resnet,
        'dataset': IDADataloader
    },
    # dense
    'dense-c': {
        'n_classes': 40,
        'n_features': 256,
        'data_conf': {
            'target': 'dense/caltech256',
            'n_base': 40,  # 1 base
            'n_incr': 0,  # 0 incremental
        },
        'network': wide_resnet,
        'dataset': IncrementalDataloader
    }
}


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


def get_config(name):
    assert name in config, "Configuration name not found."
    config[name]['data_conf']['transform'], config[name]['data_conf']['augmentation'] = get_transform(name)
    return config[name]
