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
        self.bn1 = bn(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5)
        self.bn2 = bn(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.bn3 = bn(128)
        self.conv3_drop = nn.Dropout2d()
        self.init_params()

        self.classifier = SVHN_Class_classifier(n_classes)
        self.dom_discr = SVHN_Domain_classifier()

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
        input = input.expand(input.data.shape[0], 3, 28, 28)
        x = F.relu(self.bn1(self.conv1(input)))
        x = F.max_pool2d(x, 3, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 3, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv3_drop(x)

        return x.view(-1, 128 * 3 * 3)

    def predict(self, x):
        return self.classifier(x)

    def discriminate_domain(self, x, lam):
        return self.dom_discr(x, lam)


class SVHN_Class_classifier(nn.Module):

    def __init__(self, n_classes=10):
        super(SVHN_Class_classifier, self).__init__()
        self.fc1 = nn.Linear(128 * 3 * 3, 3072)
        self.bn1 = nn.BatchNorm1d(3072)
        self.fc2 = nn.Linear(3072, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, n_classes)

    def forward(self, input):
        logits = F.relu(self.bn1(self.fc1(input)))
        logits = F.dropout(logits)
        logits = F.relu(self.bn2(self.fc2(logits)))
        logits = self.fc3(logits)

        return logits


class SVHN_Domain_classifier(nn.Module):

    def __init__(self):
        super(SVHN_Domain_classifier, self).__init__()
        self.fc1 = nn.Linear(128 * 3 * 3, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 1)

    def forward(self, input, lam):
        input = GRL(input, lam)
        logits = F.relu(self.bn1(self.fc1(input)))
        logits = F.dropout(logits)
        logits = F.relu(self.bn2(self.fc2(logits)))
        logits = F.dropout(logits)
        logits = self.fc3(logits)

        return logits


def svhn_net(pretrained=None, num_classes=10):
    return SVHN_net(num_classes)


def svhn_net_dial(pretrained=None, num_classes=10):
    return SVHN_net(num_classes, dial=True)
