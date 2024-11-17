import torch
import torch.nn as nn



class NormalHead(nn.Module):
    def __init__(self, input_dim1=32, output_dim1=32):
        super(NormalHead, self).__init__()
        self.fc1 = nn.Linear(input_dim1, output_dim1)
        self.fc2 = nn.Linear(output_dim1, 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(32)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.sigmoid(self.fc2(x))

        return x