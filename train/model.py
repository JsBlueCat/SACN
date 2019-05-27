import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
sys.path.append("/home/cery/workspace/darts-new/SACN")
from train.operations import *
from utils import drop_path


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight.data)
        # nn.init.constant(m.bias, 0.1)


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        print(C_prev_prev, C_prev, C)

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i+1]]
            op1 = self._ops[2*i]
            op2 = self._ops[2*i+1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class Network_SACN_1_fcn(nn.Module):
    def __init__(self):
        super(Network_SACN_1_fcn, self).__init__()
        # backend
        # self.pre_layer = nn.Sequential(
        #     ConvReLU(3, 16, kernel_size=3, stride=2, padding=0),
        #     ConvReLU(16, 32, kernel_size=3, stride=2, padding=0),
        #     ConvReLU(32, 64, kernel_size=3, stride=2, padding=0),
        #     ConvReLU(64, 128, kernel_size=2, stride=1, padding=0)
        # )
        # backend
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),  # PReLU1
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool1
            nn.Conv2d(16, 32, kernel_size=3, stride=1),  # conv2
            nn.PReLU(),  # PReLU2
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2,
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # conv3
            nn.PReLU()  # PReLU3
        )
        self.face_cls = nn.Conv2d(64, 1, kernel_size=1)
        self.bbox_reg = nn.Conv2d(64, 4, kernel_size=1)
        self.rig_cls = nn.Conv2d(64, 1, kernel_size=1)

        # weight initiation with xavier
        self.apply(weights_init)

    def forward(self, x):
        out = self.pre_layer(x)
        face = F.sigmoid(self.face_cls(out))
        bbox = self.bbox_reg(out)
        rip = F.sigmoid(self.rig_cls(out))
        return face, bbox, rip


class Network_SACN_1(nn.Module):

    def __init__(self, C, layers, genotype):
        super(Network_SACN_1, self).__init__()
        self._layers = layers

        stem_multiplier = 3
        C_curr = stem_multiplier*C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev,
                        C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr

        # self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Conv2d(C_prev, C_prev, kernel_size=6, stride=1)
        '''
        detect face and non-face
        '''
        self.classifier_face = nn.Conv2d(C_prev, 1, kernel_size=1, stride=1)
        '''
        bounding box regression that have 3 parameters
        a b w as shown in paper SACN
        '''
        self.bounding_box_regression = nn.Conv2d(
            C_prev, 3, kernel_size=1, stride=1)
        '''
        the RIP rotate
        '''
        self.classifier_rip = nn.Conv2d(C_prev, 1, kernel_size=1, stride=1)

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
        # out = self.global_pooling(s1)
        print(s1.size())
        out = self.global_conv(s1)
        print(out.size())
        face_prob = torch.sigmoid(
            self.classifier_face(out))
        bounding_box = self.bounding_box_regression(out)
        rip = torch.sigmoid(self.classifier_rip(out))
        return face_prob, bounding_box, rip


class Network_SACN_2(nn.Module):

    def __init__(self, C, layers, genotype):
        super(Network_SACN_2, self).__init__()
        self._layers = layers

        stem_multiplier = 3
        C_curr = stem_multiplier*C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev,
                        C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        '''
        detect face and non-face
        '''
        self.classifier_face = nn.Linear(C_prev, 1)
        '''
        bounding box regression that have 3 parameters
        a b w as shown in paper SACN
        '''
        self.bounding_box_regression = nn.Linear(C_prev, 3)
        '''
        the RIP rotate
        '''
        self.classifier_rip = nn.Linear(C_prev, 3)

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
        out = self.global_pooling(s1)
        face_prob = torch.sigmoid(
            self.classifier_face(out.view(out.size(0), -1)))
        bounding_box = self.bounding_box_regression(out.view(out.size(0), -1))
        rip = F.softmax(self.classifier_rip(out.view(out.size(0), -1)), dim=1)
        return face_prob, bounding_box, rip


class Network_SACN_3(nn.Module):

    def __init__(self, C, layers, genotype):
        super(Network_SACN_3, self).__init__()
        self._layers = layers

        stem_multiplier = 3
        C_curr = stem_multiplier*C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev,
                        C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        '''
        detect face and non-face
        '''
        self.classifier_face = nn.Linear(C_prev, 1)
        '''
        bounding box regression that have 3 parameters
        a b w as shown in paper SACN
        '''
        self.bounding_box_regression = nn.Linear(C_prev, 3)
        '''
        the RIP rotate regression
        '''
        self.classifier_rip = nn.Linear(C_prev, 1)

    def forward(self, input):
        _n, _c, h, w = input.size()
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
        out = self.global_pooling(s1)
        face_prob = torch.sigmoid(
            self.classifier_face(out.view(out.size(0), -1)))
        bounding_box = self.bounding_box_regression(out.view(out.size(0), -1))
        # logits regression
        rip = self.classifier_rip(out.view(out.size(0), -1))
        return face_prob, bounding_box, rip
