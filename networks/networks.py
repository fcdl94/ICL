'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import math
import torch.utils.model_zoo as model_zoo
from .rev_grad import grad_reverse as GRL
from .block import *
from torch.nn import init
from torchvision import models

model_urls = {
    'resnet18': 'http://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'http://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'http://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'http://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'http://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class ResNet(nn.Module):

    def __init__(self, block, layers, pretrained=None, num_classes=1000, zero_init_residual=False, bottleneck=True, bottleneck_dim=256):
        super(ResNet, self).__init__()
        self.inplanes = 64

        self.dial = False
        self.bn = nn.BatchNorm2d

        self.conv1 = pretrained.conv1 if pretrained is not None else nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = pretrained.bn1 if pretrained is not None else self.bn(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = pretrained.layer1 if pretrained is not None else self._make_layer(block, 64, layers[0])
        self.layer2 = pretrained.layer2 if pretrained is not None else self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = pretrained.layer3 if pretrained is not None else self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = pretrained.layer4 if pretrained is not None else self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        n_features_in = 512*block.expansion
        self.use_bottleneck = False

        if bottleneck:
            self.use_bottleneck = True
            self.bottleneck = nn.Linear(n_features_in, bottleneck_dim)
            n_features_in = bottleneck_dim

        self.fc = nn.Linear(n_features_in, num_classes)

        self.domain_discriminator = nn.Sequential(nn.Linear(n_features_in, 1024),
                                                  nn.ReLU(),
                                                  nn.Linear(1024, 1024),
                                                  nn.ReLU(),
                                                  nn.Linear(1024, 1))

        self.bottleneck.apply(init_weights)
        self.fc.apply(init_weights)
        self.domain_discriminator.apply(init_weights)

        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d) or isinstance(m, DAL):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            # Zero-initialize the last BN in each residual branch,
            # so that the residual branch starts with zeros, and each residual block behaves like an identity.
            # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
            if zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        nn.init.constant_(m.bn3.weight, 0)
                    elif isinstance(m, BasicBlock):
                        nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                self.bn(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dial=self.dial))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dial=self.dial))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if self.use_bottleneck:
            x = self.bottleneck(x)

        return x, x  # here logits and feats are the same! (we classify on only one FC)

    def predict(self, x):
        x = self.fc(x)
        return x

    def discriminate_domain(self, x, const):
        assert self.domain_discriminator is not None, "Calling discriminate_domain without enabling rev_grad"
        x = GRL(x, const)
        x = self.domain_discriminator(x)
        return x, x

    def set_domain(self, domain):
        for mod in self.modules():
            if isinstance(mod, DAL):
                mod.set_domain(domain)

    def set_source(self):
        self.set_domain(0)

    def set_target(self):
        self.set_domain(1)


class CifarResNet(nn.Module):
    """
    ResNet optimized for the Cifar Dataset, as specified in
    https://arxiv.org/abs/1512.03385.pdf
    """

    def __init__(self, block=BasicBlock, depth=32, num_classes=100, channels=3, dial=False, revgrad=False):

        super(CifarResNet, self).__init__()

        # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6

        self.dial = dial
        if not dial:
            bn = nn.BatchNorm2d
        else:
            bn = DAL

        self.revgrad = revgrad

        self.num_classes = num_classes

        self.conv_1_3x3 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = bn(16)

        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1, dial)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2, dial)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2, dial, last=True)
        self.avgpool = nn.AvgPool2d(8)
        self.linear = nn.Linear(64, num_classes)

        if revgrad:
            self.domain_discriminator = nn.Sequential(nn.Linear(64, 1024),
                                                      nn.ReLU(),
                                                      nn.Linear(1024, 1024),
                                                      nn.ReLU(),
                                                      nn.Linear(1024, 1))
        else:
            self.domain_discriminator = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                init.kaiming_normal_(m.weight, nonlinearity='relu')
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, DAL):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, math.sqrt(1. / 64.))
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dial=False, last=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)

        layers = [block(self.inplanes, planes, stride, downsample, dial=dial)]

        self.inplanes = planes * block.expansion
        if last:
            for i in range(1, blocks - 1):
                layers.append(block(self.inplanes, planes, dial=dial))
            layers.append(block(self.inplanes, planes, dial=dial, relu=False))

        else:
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, dial=dial))

        return nn.Sequential(*layers)

    def set_domain(self, domain):
        for mod in self.modules():
            if isinstance(mod, DAL):
                mod.set_domain(domain)

    def set_source(self):
        self.set_domain(0)

    def set_target(self):
        self.set_domain(1)

    def forward(self, x):

        x = self.conv_1_3x3(x)

        x = self.bn_1(x)
        x = F.relu(x)

        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x, x  # here logits and feats are the same! (we classify on only one FC)

    def predict(self, x):
        out = self.linear(x)
        return out

    def discriminate_domain(self, x, const):
        assert self.domain_discriminator is not None, "Calling discriminate_domain without enabling rev_grad"
        x = GRL(x, const)
        x = self.domain_discriminator(x)
        return x


class WideResNet(nn.Module):
    def __init__(self, resnet_block, widening_factor=4, num_classes=1000, dial=False, revgrad=False):
        super(WideResNet, self).__init__()

        self.dial = dial
        if not dial:
            bn = nn.BatchNorm2d
        else:
            bn = DAL

        self.block = nn.Conv2d
        self.inplanes = 16

        self.conv1 = self.block(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = bn(self.inplanes)

        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(resnet_block, 64, widening_factor, dial=dial, stride=2)
        self.layer2 = self._make_layer(resnet_block, 128, widening_factor, dial=dial, stride=2)
        self.layer3 = self._make_layer(resnet_block, 256, widening_factor, dial=dial, stride=2, last=True)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(256, num_classes)
        self.index = 0

        self.revgrad = revgrad

        if revgrad:
            self.domain_discriminator = nn.Sequential(nn.Linear(256, 1024),
                                                      nn.ReLU(),
                                                      nn.Linear(1024, 1024),
                                                      nn.ReLU(),
                                                      nn.Linear(1024, 1))
        else:
            self.domain_discriminator = None

    def _make_layer(self, block, planes, blocks, dial=False, stride=1, last=False):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = DownsampleD(self.inplanes, planes*block.expansion, stride,
                                     nn.BatchNorm2d if not dial else DAL)

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion

        if last:
            for i in range(1, blocks - 1):
                layers.append(block(self.inplanes, planes, dial=dial))
            layers.append(block(self.inplanes, planes, dial=dial, relu=False))

        else:
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, dial=dial))

        return nn.Sequential(*layers)

    def set_domain(self, domain):
        for mod in self.modules():
            if isinstance(mod, DAL):
                mod.set_domain(domain)

    def set_source(self):
        self.set_domain(0)

    def set_target(self):
        self.set_domain(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x, x  # here logits and feats are the same! (we classify on only one FC)

    def predict(self, x):
        return self.fc(x)

    def discriminate_domain(self, x, const):
        assert self.domain_discriminator is not None, "Calling discriminate_domain without enabling rev_grad"
        x = GRL(x, const)
        x = self.domain_discriminator(x)
        return x


def cifar_resnet(pretrained=None, num_classes=1000):
    model = CifarResNet(num_classes=num_classes, revgrad=True)
    if pretrained is not None:
        state_dict = torch.load(pretrained)
        model.load_state_dict(state_dict['network'])
        print(f"Model pretrained loaded {pretrained}")

    return model


def cifar_resnet_dial(pretrained=None, num_classes=1000):
    model = CifarResNet(num_classes=num_classes, dial=True)
    if pretrained is not None:
        state_dict = torch.load(pretrained)
        model.load_state_dict(state_dict['network'])
        print(f"Model pretrained loaded {pretrained}")

    return model


def cifar_resnet_revgrad(pretrained=None, num_classes=1000):
    model = CifarResNet(num_classes=num_classes,)
    if pretrained is not None:
        state_dict = torch.load(pretrained)
        model.load_state_dict(state_dict['network'])
        print(f"Model pretrained loaded {pretrained}")

    return model


def wide_resnet(pretrained=None, num_classes=1000):
    model = WideResNet(BasicBlock, num_classes=num_classes)
    if pretrained is not None:
        state_dict = torch.load(pretrained)
        model.load_state_dict(state_dict['network'])
        print(f"Model pretrained loaded {pretrained}")

    return model


def wide_resnet_dial(pretrained=None, num_classes=1000):
    model = WideResNet(BasicBlock, num_classes=num_classes, dial=True)
    if pretrained is not None:
        state_dict = torch.load(pretrained)
        model.load_state_dict(state_dict['network'])
        print(f"Model pretrained loaded {pretrained}")

    return model


def wide_resnet_revgrad(pretrained=None, num_classes=1000):
    model = WideResNet(BasicBlock, num_classes=num_classes, revgrad=True)
    if pretrained is not None:
        state_dict = torch.load(pretrained)
        model.load_state_dict(state_dict['network'])
        print(f"Model pretrained loaded {pretrained}")

    return model


def resnet18(pretrained=None, num_classes=1000, bottleneck=True, bottleneck_dim=256):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes (int): Number of classes of the system
    """
    if pretrained is not None:
        pre_model = models.resnet18(True)
    else:
        pre_model = None

    model = ResNet(BasicBlock, [2, 2, 2, 2], pretrained=pre_model,
                   bottleneck=bottleneck, bottleneck_dim=bottleneck_dim, num_classes=num_classes)

    return model


def resnet34(pretrained=None, num_classes=1000, bottleneck=True, bottleneck_dim=256):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes (int): Number of classes of the system
    """
    if pretrained is not None:
        pre_model = models.resnet34(True)
    else:
        pre_model = None

    model = ResNet(BasicBlock, [3, 4, 6, 3], pretrained=pre_model,
                   bottleneck=bottleneck, bottleneck_dim=bottleneck_dim, num_classes=num_classes)

    return model


def resnet50(pretrained=None, num_classes=1000, bottleneck=True, bottleneck_dim=256):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes (int): Number of classes of the system
    """
    if pretrained is not None:
        pre_model = models.resnet50(True)
    else:
        pre_model = None

    model = ResNet(Bottleneck, [3, 4, 6, 3], pretrained=pre_model,
                   bottleneck=bottleneck, bottleneck_dim=bottleneck_dim, num_classes=num_classes)

    return model


def resnet50_dial(pretrained=None, num_classes=1000):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes (int): Number of classes of the system
    """
    raise NotImplementedError
