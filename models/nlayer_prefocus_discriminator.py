from typing import List
import torch
import torch.nn as nn
import functools
from .simple_resnet import ResNet18
from .pregan_networks import ResnetBlock


class NLayerPreFocusDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, pretrained_model: str, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, pretrained_class:bool=True):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerPreFocusDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.netPre: nn.Module = ResNet18(num_outputs=128)
        self.netPre.load_state_dict(torch.load(pretrained_model))
        self.feature_layer = 2
        self.pretrained_class = pretrained_class
        self.netPre.eval()
        for parameters in self.netPre.parameters():
            parameters.requires_grad = False

        kw = 4
        padw = 1
        sequence_1 = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]

        # Class Activation Map
        ActMap = []
        ActMap += [
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, ndf, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(ndf),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(ndf),
            nn.ReLU(True),

            ResnetBlock(ndf, use_bias=False),
            nn.Conv2d(ndf, ndf, kernel_size=1, stride=1, bias=True),
            nn.ReLU(True),
        ]

        sequence_2 = []
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence_2 += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence_2 += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence_2 += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.sequence_1 = nn.Sequential(*sequence_1)
        self.actmap = nn.Sequential(*ActMap)
        self.sequence_2 = nn.Sequential(*sequence_2)
        if self.pretrained_class:
            self.pext = nn.Sequential(
                            nn.Linear(512, 512),
                            nn.ReLU(),
                            nn.Linear(512, 900)
                        )

    def forward(self, input, heatmap: List[torch.Tensor]=None):
        x = self.sequence_1(input)

        features = []
        self.netPre(input, features=features)
        feature = features[self.feature_layer]

        act = self.actmap(feature)
        x = x * act
        if heatmap is not None:
            cam = torch.mean(act, dim=1)
            heatmap.append(cam)

        out = self.sequence_2(x)
        out = out.view(out.size(0), -1)
        if self.pretrained_class:
            out2 = self.pext(features[-1])
            out = torch.cat([out, out2], dim=1)
        return out