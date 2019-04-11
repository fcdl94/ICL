import torch
import torch.nn as nn
import torch.nn.functional as F
from .rev_grad import grad_reverse as GRL


class GTSRB_net(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=5)
        self.conv2 = nn.Conv2d(96, 144, kernel_size=3)
        self.conv3 = nn.Conv2d(144, 256, kernel_size=5)
        self.max_pool = nn.MaxPool2d(2, stride=2)

        self.classifier = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, n_classes))
        self.domain_discr = nn.Sequential(nn.Linear(256, 1024), nn.ReLU(),
                                          nn.Linear(1024, 1024), nn.ReLU(),
                                          nn.Linear(1024, 1))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max_pool(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)
        x = F.relu(self.conv3(x))
        x = self.max_pool(x)
        x = x.view(x.size(0), -1)
        return x

    def predict(self, x):
        x = self.classifier(x)
        return x

    def discriminate_domain(self, x):
        x = GRL(x)
        x = self.domain_discr(x)
        return x


if __name__=="__main__":
    net = GTSRB_net(10)
    x = torch.randn(2, 3, 32, 32)
    x = x.to("cuda")
    net.to("cuda")

    x = net.forward(x)
    print(net.predict(x).shape)
    print(net.discriminate_domain(x).shape)
