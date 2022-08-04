import torch.nn.functional as F 
from torch import nn
from pg_modules import blocks
from pg_modules.networks_fastgan import FastganSynthesis


class Encoder(nn.Module):
    def __init__(self, ngf=128, img_resolution: int=256, num_outputs:int=256):
        super().__init__()
        nfc_multi = {2: 16, 4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5,
                     512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)

        assert img_resolution in nfc
        self.init_layer = nn.Conv2d(3, nfc[img_resolution], kernel_size=7, stride=1, padding=3)
        DownBlock = blocks.DownBlock
        self.down_layers = nn.ModuleList()

        while img_resolution > 2:
            down = DownBlock(nfc[img_resolution], nfc[img_resolution//2])
            self.down_layers.append(down)
            img_resolution = img_resolution // 2
        
        self.out_layer = nn.Linear(nfc[img_resolution], num_outputs)
        
    def forward(self, input):
        x = self.init_layer(input)
        for _, module in enumerate(self.down_layers):
            x = module(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.out_layer(x)
