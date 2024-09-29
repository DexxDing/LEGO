import pdb

import torch
import torch.nn as nn

class NormalHead(nn.Module):
    def __init__(self, in_channel=32, ratios=[1, 2], kernel_sizes=[1, 1, 1]):
        super(NormalHead, self).__init__()
        self.ratios = ratios
        self.kernel_sizes = kernel_sizes

        self.build_layers(in_channel)
        
    def build_layers(self, in_channel):
        ratio_1, ratio_2 = self.ratios
        self.conv1 = nn.Conv1d(in_channel, in_channel // ratio_2,
                               self.kernel_sizes[0])
        # self.bn1 = nn.BatchNorm1d(in_channel // ratio_2)
        # self.conv2 = nn.Conv1d(in_channel // ratio_1, in_channel // ratio_2,
        #                        self.kernel_sizes[1], 1, self.kernel_sizes[1] // 2)
        self.bn2 = nn.BatchNorm1d(in_channel // ratio_2)
        self.conv3 = nn.Conv1d(in_channel // ratio_2, 1,
                               self.kernel_sizes[2])
        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # self.bns = [self.bn1, self.bn2]
        self.bns = [self.bn2]

    def forward(self, x):
        '''
        x: BN * C * T
        return BN * C // 64 * T and BN * 1 * T
        '''
        outputs = []
        #  x: [5, 512, 67]
        x = self.conv1(x)
        x = self.act(x)
        #  conv1: [5, 32, 67]
        outputs.append(x)
        # x = self.conv2(self.act(self.bn1(x)))
        # #  conv2:  [5, 16, 67]
        # outputs.append(x)
        x = self.sigmoid(self.conv3(self.act(self.bn2(x))))
        #  conv3: [5, 1, 67]
        outputs.append(x)
        #  output0: [5, 32, 67], output1: [5, 16, 67], output2: []

        # log_message_grad = f'conv1 grad: {self.conv1.weight.grad} \n conv2 grad: {self.conv2.weight.grad} \n conv3 grad: {self.conv3.weight.grad}'
        return outputs
