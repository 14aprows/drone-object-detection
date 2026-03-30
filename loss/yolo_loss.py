import torch 
import torch.nn as nn
from utils.iou import iou

class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, predictions, target, anchors):
        obj = target[..., 0] == 1
        no_obj = target[..., 0] == 0

        no_obj_loss = self.bce(
            (predictions[..., 0:1][no_obj]),
            (target[..., 0:1][no_obj])
        )

        anchors = anchors.reshape(1, 3, 1, 1, 2)
        pred_xy = self.sigmoid(predictions[..., 1:3])
        pred_wh = torch.exp(predictions[..., 3:5]) * anchors
        box_preds = torch.cat((pred_xy, pred_wh), dim=-1)

        ious = iou(box_preds[obj], target[..., 1:5][obj])

        object_loss = self.mse(
            self.sigmoid(predictions[..., 0:1][obj]),
            ious * target[..., 0:1][obj]
        )

        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / anchors)
        )
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        class_loss = self.cross_entropy(
            (predictions[..., 5:][obj]),
            (target[..., 5][obj].long())
        )


        return (
            box_loss +
            object_loss +
            no_obj_loss +
            class_loss
        )