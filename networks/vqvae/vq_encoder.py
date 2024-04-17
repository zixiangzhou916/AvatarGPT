import os, sys
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.vqvae.layers import *
from networks.vqvae.resnet import Resnet1D
from networks.utils.positional_encoding import PositionalEncoding
import importlib

def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
class VQEncoder(nn.Module):
    def __init__(self, input_size, channels, n_down, **kwargs):
        super(VQEncoder, self).__init__()
        assert len(channels) == n_down
        layers = [
            nn.Conv1d(input_size, channels[0], 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(channels[0]),
        ]

        for i in range(1, n_down):
            layers += [
                nn.Conv1d(channels[i-1], channels[i], 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock(channels[i]),
            ]
        self.main = nn.Sequential(*layers)
        # self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        # self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        # print(outputs.shape)
        return outputs  # [bs, nframes / 4, 1024]
    
class VQEncoderHML(nn.Module):
    def __init__(
        self, 
        input_emb_width = 263,
        output_emb_width = 512,
        down_t = 2,
        stride_t = 2,
        width = 512,
        depth = 3,
        dilation_growth_rate = 3,
        activation='relu',
        norm=None, 
        **kwargs
    ):
        super(VQEncoderHML, self).__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        
        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, inputs):
        x = inputs.permute(0, 2, 1)
        y = self.model(x)
        return y.permute(0, 2, 1)