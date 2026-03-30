import torch 
from utils.iou import iou

def nms(bboxes, iou_threshold, threshold):
    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    
    bboxes_nms = []
    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes_nms.append(chosen_box)
        bboxes = [
            box for box in bboxes
            if box[0] != chosen_box[0]
            or iou(torch.tensor(box[2:]), torch.tensor(chosen_box[2:]), True) < iou_threshold
        ]    
    return bboxes_nms