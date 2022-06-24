from typing import List
import torch
import torch.nn as nn
from .basic_networks import ILN


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size

        DownBlock = []
        DownBlock += [nn.ReflectionPad2d(3),
                      nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=False),
                      nn.InstanceNorm2d(ngf),
                      nn.ReLU(True)]

        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            DownBlock += [nn.ReflectionPad2d(1),
                          nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0, bias=False),
                          nn.InstanceNorm2d(ngf * mult * 2),
                          nn.ReLU(True)]

        # Down-Sampling Bottleneck
        mult = 2**n_downsampling
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf * mult, use_bias=False)]

        # Class Activation Map
        ActMap = []
        ActMap += [
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0, bias=False),
            ILN(128),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, ngf * mult, kernel_size=3, stride=1, padding=0, bias=False),
            ILN(ngf * mult),
            nn.ReLU(True),
            ResnetBlock(ngf * mult, use_bias=False),
            nn.Conv2d(ngf * mult, ngf * mult, kernel_size=1, stride=1, bias=True),
        ]

        # Up-Sampling
        UpBlock = []
        for i in range(n_blocks):
            UpBlock += [ResnetBlock(ngf * mult, use_bias=False)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            UpBlock += [nn.Upsample(scale_factor=2, mode='nearest'),
                         nn.ReflectionPad2d(1),
                         nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0, bias=False),
                         ILN(int(ngf * mult / 2)),
                         nn.ReLU(True)]

        UpBlock += [nn.ReflectionPad2d(3),
                     nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=False),
                     nn.Tanh()]

        self.DownBlock = nn.Sequential(*DownBlock)
        self.ActMap = nn.Sequential(*ActMap)
        self.UpBlock = nn.Sequential(*UpBlock)

    def forward(self, input: torch.Tensor, feature: torch.Tensor, features:List=None, max_layer: int=-1): 
        x: torch.Tensor = input
        if features is None:
            x = self.DownBlock(x)
        else:
            for _, layer in enumerate(self.DownBlock):
                x = layer(x)
                features.append(x)
                if max_layer >= 0 and len(features) > max_layer:
                    return

        actMap = self.ActMap(feature)
        x = x * actMap
        heatmap = torch.mean(actMap, dim = 1)
        if features is not None:
            features.append(x)
            if max_layer >= 0 and len(features) > max_layer:
                return

        if features is None:
            x = self.UpBlock(x)
        else:
            for _, layer in enumerate(self.UpBlock):
                x = layer(x)
                features.append(x)
                if max_layer >= 0 and len(features) > max_layer:
                    return

        return x, heatmap


class ResnetBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out