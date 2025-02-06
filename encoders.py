import torch
import torch.nn as nn
import torch.nn.functional as F
from residual import ResidualStack

class Encoder(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens):
        """
        Returns a feature map of shape [B, num_hiddens, H/4, W/4] after 2 strided convs.
        """
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=num_hiddens // 2, 
            kernel_size=4, 
            stride=2, 
            padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=num_hiddens // 2, 
            out_channels=num_hiddens, 
            kernel_size=4, 
            stride=2, 
            padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=num_hiddens, 
            out_channels=num_hiddens, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        self.residual_stack = ResidualStack(num_hiddens, num_residual_layers, num_residual_hiddens)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.residual_stack(x)
        return x