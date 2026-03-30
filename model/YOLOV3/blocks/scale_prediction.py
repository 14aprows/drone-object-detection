import torch.nn as nn

class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.pred = nn.Sequential(
            nn.Conv2d(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * in_channels),
            nn.LeakyReLU(0.1),

            nn.Conv2d(
                2 * in_channels,
                3 * (num_classes + 5),
                kernel_size=1
            ),
        )

    def forward(self, x):
        x = self.pred(x)

        x = x.reshape(
            x.shape[0],     
            3,               
            self.num_classes + 5,
            x.shape[2],
            x.shape[3],
        )

        x = x.permute(0, 1, 3, 4, 2)
        return x