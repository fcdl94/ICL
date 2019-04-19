import torch
import torch.nn as nn
import torch.nn.functional as F
from .rev_grad import grad_reverse as GRL
from .dial import DomainAdaptationLayer as DAL

class GTSRB_net(nn.Module):
    def __init__(self, n_classes=10, dial=False):
        super().__init__()
        if not dial:
            bn = nn.BatchNorm2d
        else:
            bn = DAL

        self.conv1 = nn.Conv2d(3, 96, kernel_size=5)
        self.bn1 = bn(96)
        self.conv2 = nn.Conv2d(96, 144, kernel_size=3)
        self.bn2 = bn(144)
        self.conv3 = nn.Conv2d(144, 256, kernel_size=5)
        self.bn3 = bn(256)
        self.max_pool = nn.MaxPool2d(2, stride=2)

        self.fc1 = nn.Linear(1024, 512)

        self.classifier = nn.Linear(512, n_classes)
        self.domain_discr = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(),
                                          nn.Linear(1024, 1024), nn.ReLU(),
                                          nn.Linear(1024, 1))

    def set_domain(self, domain):
        for mod in self.modules():
            if isinstance(mod, DAL):
                mod.set_domain(domain)

    def set_source(self):
        self.set_domain(0)

    def set_target(self):
        self.set_domain(1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.max_pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.max_pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.max_pool(x)

        feat = x.view(x.size(0), -1)
        logits = self.fc1(feat)

        return logits, feat

    def predict(self, logits):
        return self.classifier(logits)

    def discriminate_domain(self, feat, lam):
        x = GRL(feat, lam)
        x = self.domain_discr(x)
        return x


def gtsrb_net(pretrained=None, num_classes=43):
    return GTSRB_net(num_classes)


def gtsrb_net_dial(pretrained=None, num_classes=43):
    return GTSRB_net(num_classes, dial=True)


if __name__=="__main__":
    net = GTSRB_net(10)
    x = torch.randn(2, 3, 32, 32)
    x = x.to("cuda")
    net.to("cuda")

    x = net.forward(x)
    print(net.predict(x).shape)
    print(net.discriminate_domain(x, 0.1).shape)
