import torch
import torch.nn as nn

from model.YOLOV3.blocks.cnn_block import CNNBlock
from model.YOLOV3.blocks.residual_block import ResidualBlock
from model.YOLOV3.blocks.scale_prediction import ScalePrediction


class YOLOv3(nn.Module):
    """
    Simplified YOLOv3 Architecture
    Backbone: Darknet53-like
    Detection: 3 scales
    """

    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()

        # =========================
        # Backbone (Darknet53)
        # =========================

        self.backbone = nn.ModuleList([

            CNNBlock(in_channels, 32, kernel_size=3, stride=1, padding=1),

            CNNBlock(32, 64, kernel_size=3, stride=2, padding=1),
            ResidualBlock(64, num_repeats=1),

            CNNBlock(64, 128, kernel_size=3, stride=2, padding=1),
            ResidualBlock(128, num_repeats=2),

            CNNBlock(128, 256, kernel_size=3, stride=2, padding=1),
            ResidualBlock(256, num_repeats=8),  # route connection

            CNNBlock(256, 512, kernel_size=3, stride=2, padding=1),
            ResidualBlock(512, num_repeats=8),  # route connection

            CNNBlock(512, 1024, kernel_size=3, stride=2, padding=1),
            ResidualBlock(1024, num_repeats=4),
        ])

        # =========================
        # Detection Heads
        # =========================

        self.scale1 = ScalePrediction(1024, num_classes)
        self.scale2 = ScalePrediction(1536, num_classes)
        self.scale3 = ScalePrediction(1792, num_classes)

        # upsample layer
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):

        outputs = []
        route_connections = []

        # =========================
        # Backbone Forward
        # =========================

        for layer in self.backbone:

            x = layer(x)

            # save feature maps for FPN
            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

        # =========================
        # Scale 1 (13x13)
        # =========================

        outputs.append(self.scale1(x))

        # =========================
        # Scale 2 (26x26)
        # =========================

        x = self.upsample(x)
        x = torch.cat([x, route_connections[-1]], dim=1)

        outputs.append(self.scale2(x))

        # =========================
        # Scale 3 (52x52)
        # =========================

        x = self.upsample(x)
        x = torch.cat([x, route_connections[-2]], dim=1)

        outputs.append(self.scale3(x))

        return outputs