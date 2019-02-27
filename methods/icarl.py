from .common import CifarResNet
import torch.nn as nn
import torch.optim as optim

LR = 0.01
MEM_SIZE = 100
DECAY = 0.9

class ICarl:

    def __init__(self, device):
        self.network = CifarResNet().to(device)
        self.loss = nn.BCEWithLogitsLoss(size_average=True)

        self.lr_init = LR
        self.mem_size = MEM_SIZE

        self.prototypes = None

    def fit(self):
        pass

    def incremental_fit(self, X, y):
        new_lr = self.lr_init
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.network.parameters()), lr=new_lr, momentum=0.9,
                              weight_decay=DECAY, nesterov=False)
        # update_model_representation(X,P,model)
        # ex_per_class = self.mem_size / t
        # for each old class, REDUCE_EX_SET(P[y], ex_per_class)
        # for each new class, CREATE_EX_SET(X[y], ex_per_class, model)

    def test(self):
        pass
