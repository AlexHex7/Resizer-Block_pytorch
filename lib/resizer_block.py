import torch
from torch import nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, 1, 1),
            nn.BatchNorm2d(in_c),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_c, in_c, 3, 1, 1),
            nn.BatchNorm2d(in_c),
        )
    
    def forward(self, x):
        return self.layers(x) + x


class ResizerBlock(nn.Module):
    def __init__(self, in_c, output_size=(64, 64), n=16, r=1, sample_mode='bilinear', align_corners=True):
        super().__init__()
        self.output_size = output_size
        self.sample_mode = sample_mode
        self.align_corners = align_corners

        self.head = nn.Sequential(
            nn.Conv2d(in_c, n, 7, 1, 3),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(n, n, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(n),
        )

        layer_list = []
        for _ in range(r):
            layer_list.append(ResBlock(n))
        self.body = nn.Sequential(
            *layer_list,
            nn.Conv2d(n, n, 3, 1, 1),
            nn.BatchNorm2d(n)
        )

        self.tail = nn.Sequential(
            nn.Conv2d(n, in_c, 7, 1, 3),
        )
    
    def forward(self, x):
        x_branch = F.interpolate(x, size=self.output_size, mode=self.sample_mode, align_corners=self.align_corners)
        output = F.interpolate(self.head(x), size=self.output_size, mode=self.sample_mode, align_corners=self.align_corners)

        return self.tail(self.body(output) + output) + x_branch


if __name__ == '__main__':
    x = torch.randn(1, 3, 200, 200)

    net = ResizerBlock(in_c=3, output_size=(64, 64), n=16, r=1, sample_mode='bilinear', align_corners=True)
    print(net(x).size())