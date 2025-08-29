import torch
import torch.nn as nn
from torch.nn import init
from einops import rearrange, repeat
import torch_sparse


class AGCA(nn.Module):
    def __init__(self, in_channel, ratio):
        super(AGCA, self).__init__()
        hide_channel = in_channel // ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channel, hide_channel, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(2)
        # self.A2 = nn.Parameter(torch.FloatTensor(torch.zeros((in_channel, 233))), requires_grad=True)
        # init.constant_(self.A2, 1e-6)

        # Convolutions to transform and generate A1
        self.conv2 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(hide_channel, in_channel, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, adj):

        y = self.avg_pool(x.unsqueeze(2).unsqueeze(3))
        y = self.conv1(y)
        B, C, _, _ = y.size()

        y = y.flatten(2).transpose(1, 2)

        A1 = self.softmax(self.conv2(y))
        A1 = A1.squeeze(1)
        A2 = torch.zeros((x.size(0), 233), device=x.device, requires_grad=True)
        init.constant_(A2, 1e-6)
        A0 = adj.to_dense().to(x.device)
        A = torch.matmul(A0, A1)
        A += A2
        y = self.relu(self.conv3(y))
        y = y.transpose(1, 2).view(-1, C, 1, 1)
        y = self.sigmoid(self.conv4(y))
        y = y.squeeze(3).transpose(1, 2)

        return y



