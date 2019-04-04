import torch
from torch import nn, Tensor


class DomainAdaptationLayer(nn.Module):
    def __init__(self, planes):
        super(DomainAdaptationLayer, self).__init__()

        self.bn_source = nn.BatchNorm2d(planes)
        nn.init.constant_(self.bn_source.weight, 1)
        nn.init.constant_(self.bn_source.bias, 0)
        self.bn_source.weight.requires_grad = False
        self.bn_source.bias.requires_grad = False

        self.bn_target = nn.BatchNorm2d(planes)
        nn.init.constant_(self.bn_target.weight, 1)
        nn.init.constant_(self.bn_target.bias, 0)
        self.bn_target.weight.requires_grad = False
        self.bn_target.bias.requires_grad = False

        self.weight = nn.parameter.Parameter(Tensor(planes))
        self.bias = nn.parameter.Parameter(Tensor(planes))
        self.domain = 0

    def set_domain(self, domain):
        self.domain = domain

    def forward(self, x, index=None):
        if index is None:
            index = self.domain

        if index == 0:
            out = self.bn_source(x)
        else:
            out = self.bn_target(x)

        res = self.weight.view(1, self.weight.size()[0], 1, 1) * out + self.bias.view(1, self.weight.size()[0], 1, 1)
        return res


if __name__ == '__main__':
    da = DomainAdaptationLayer(2)
    print(da)

    da.train()

    sor = torch.ones(128, 2, 4, 4)
    tar = torch.zeros(128, 2, 4, 4)
    ra = torch.randn(1, 2, 4, 4)

    da(sor, 0)
    da(tar, 1)

    da.eval()
    print(da(ra, 0))
    print(da(ra, 1))

