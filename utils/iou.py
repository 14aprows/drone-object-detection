import torch

def iou(box1, box2, is_pred=True):
    if is_pred:
        b1_x1 = box1[..., 0:1] - box1[..., 2:3] / 2
        b1_y1 = box1[..., 1:2] - box1[..., 3:4] / 2
        b1_x2 = box1[..., 0:1] + box1[..., 2:3] / 2
        b1_y2 = box1[..., 1:2] + box1[..., 3:4] / 2

        b2_x1 = box2[..., 0:1] - box2[..., 2:3] / 2
        b2_y1 = box2[..., 1:2] - box2[..., 3:4] / 2
        b2_x2 = box2[..., 0:1] + box2[..., 2:3] / 2
        b2_y2 = box2[..., 1:2] + box2[..., 3:4] / 2

        x1 = torch.max(b1_x1, b2_x1)
        y1 = torch.max(b1_y1, b2_y1)
        x2 = torch.min(b1_x2, b2_x2)
        y2 = torch.min(b1_y2, b2_y2)

        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union = box1_area + box2_area - intersection

        epsilon = 1e-6
        return intersection / (union + epsilon)

    else:
        box1 = box1.unsqueeze(0)
        if box2.dim() == 1:
            box2 = box2.unsqueeze(0)

        inter_w = torch.min(box1[:, 0], box2[:, 0]).clamp(0)
        inter_h = torch.min(box1[:, 1], box2[:, 1]).clamp(0)

        intersection = inter_w * inter_h
        box1_area = box1[:, 0] * box1[:, 1]
        box2_area = box2[:, 0] * box2[:, 1]
        union = box1_area + box2_area - intersection

        epsilon = 1e-6
        return intersection / (union + epsilon)