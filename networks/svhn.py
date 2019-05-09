"""
This network are inspired to the ones defined in https://github.com/CuthbertCai/pytorch_DANN
Credits to @CuthbertCai

"""

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .rev_grad import grad_reverse as GRL
from .dial import DomainAdaptationLayer as DAL


class SVHN_net(nn.Module):

    def __init__(self, n_classes=10, dial=False):
        super(SVHN_net, self).__init__()

        if not dial:
            bn = nn.BatchNorm2d
        else:
            bn = DAL

        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        #self.bn1 = bn(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5)
        #self.bn2 = bn(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        #self.bn3 = bn(128)
        self.conv3_drop = nn.Dropout2d()
        self.dropout = nn.Dropout()

        self.fc1 = nn.Linear(128 * 3 * 3, 3072)
        self.fc2 = nn.Linear(3072, 2048)

        self.fc3 = nn.Linear(2048, n_classes)

        self.dom_discr = SVHN_Domain_classifier()

        self.init_params()

    def init_params(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, DAL):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def set_domain(self, domain):
        for mod in self.modules():
            if isinstance(mod, DAL):
                mod.set_domain(domain)

    def set_source(self):
        self.set_domain(0)

    def set_target(self):
        self.set_domain(1)

    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = F.max_pool2d(x, 3, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 3, 2)
        x = F.relu(self.conv3(x))
        x = self.conv3_drop(x)

        feat = x.view(-1, 128 * 3 * 3)

        logits = F.relu(self.fc1(feat))
        logits = self.dropout(logits)
        logits = F.relu(self.fc2(logits))

        return logits, feat

    def predict(self, x):
        return self.fc3(x)

    def discriminate_domain(self, feat, lam):
        return self.dom_discr(feat, lam)


class SVHN_Domain_classifier(nn.Module):

    def __init__(self):
        super(SVHN_Domain_classifier, self).__init__()
        self.fc1 = nn.Linear(128 * 3 * 3, 1024)
        #self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        #self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 1)
        self.dropout = nn.Dropout()

    def forward(self, feat, lam):
        x = GRL(feat, lam)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x_ = F.relu(self.fc2(x))
        x = self.dropout(x_)
        x = self.fc3(x)

        return x, x_


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=5)

        self.fc1 = nn.Linear(48 * 4 * 4, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 10)
        self.dom_discr = Domain_classifier()

    def set_domain(self, domain):
        for mod in self.modules():
            if isinstance(mod, DAL):
                mod.set_domain(domain)

    def set_source(self):
        self.set_domain(0)

    def set_target(self):
        self.set_domain(1)

    def forward(self, input):
        x = F.max_pool2d(F.relu(self.conv1(input)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        feat = x.view(-1, 48 * 4 * 4)

        logits = F.relu(self.fc1(feat))
        logits = F.relu(self.fc2(logits))

        return logits, feat

    def predict(self, logits):
        return self.fc3(logits)

    def discriminate_domain(self, feat, lam):
        return self.dom_discr(feat, lam)


class Domain_classifier(nn.Module):

    def __init__(self):
        super(Domain_classifier, self).__init__()
        self.fc1 = nn.Linear(48 * 4 * 4, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x, constant):
        x = GRL(x, constant)
        logits_ = F.relu(self.fc1(x))
        logits = self.fc2(logits_)

        return logits, logits_


def svhn_net(pretrained=None, num_classes=10):
    return SVHN_net(num_classes)


def svhn_net_dial(pretrained=None, num_classes=10):
    return SVHN_net(num_classes, dial=True)


def lenet_net(pretrained=None, num_classes=10):
    return LeNet()
