from torch import nn
from lib.resizer_block import ResizerBlock


class ConvBNReLU(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, 1, 1),
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.layers(x)


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.resizer = ResizerBlock(in_c=1, output_size=(64, 64), n=16, r=1, sample_mode='bilinear', align_corners=True)

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            ConvBNReLU(32),
            nn.MaxPool2d(2),

            ConvBNReLU(32),
            nn.MaxPool2d(2),

            ConvBNReLU(32),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=4*4*32, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(in_features=256, out_features=10)
        )

    def forward(self, x):
        batch_size = x.size(0)

        x = self.resizer(x)

        feature = self.convs(x).view(batch_size, -1)
        output = self.fc(feature)

        return output


if __name__ == '__main__':
    import torch

    img = torch.randn(3, 1, 64, 64)
    # img = torch.randn(3, 1, 28, 28)
    net = Network()
    out = net(img)
    print(out.size())

