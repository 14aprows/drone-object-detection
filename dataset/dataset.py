import torch
import numpy as np
import os
from PIL import Image
from utils.iou import iou

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        anchors,
        image_size=416,
        grid_sizes=[13,26,52],
        num_classes=1,
        transform=None
    ):
        self.image_size = image_size
        self.transform = transform
        self.grid_sizes = grid_sizes
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.num_classes = num_classes
        self.ignore_iou_thresh = 0.5
        self.data_dir = data_dir

        self.image_files = [
            f for f in os.listdir(data_dir)
            if f.endswith(".JPEG")
        ]

        self.label_files = [
            f for f in os.listdir(data_dir)
            if f.endswith(".txt")
        ]

        self.image_label_pairs = []

        for img_file in self.image_files:
            label_file = img_file.rsplit(".",1)[0] + ".txt"
            if label_file in self.label_files:
                label_path = os.path.join(self.data_dir,label_file)
                if self.is_valid_label_file(label_path):
                    self.image_label_pairs.append(
                        (img_file,label_file)
                    )

        print("Valid pairs:",len(self.image_label_pairs))

    def __len__(self):
        return len(self.image_label_pairs)


    def __getitem__(self,idx):
        img_name,label_name = self.image_label_pairs[idx]
        img_path = os.path.join(self.data_dir,img_name)
        label_path = os.path.join(self.data_dir,label_name)

        image = np.array(
            Image.open(img_path).convert("RGB")
        )

        bboxes = np.roll(
            np.loadtxt(label_path,delimiter=" ",ndmin=2),
            4,
            axis=1
        ).tolist()

        if self.transform:
            augs = self.transform(
                image=image,
                bboxes=bboxes
            )

            image = augs["image"]
            bboxes = augs["bboxes"]

        targets = [
            torch.zeros(
                (self.num_anchors_per_scale,s,s,6)
            )
            for s in self.grid_sizes
        ]

        for box in bboxes:
            iou_anchors = iou(
                torch.tensor(box[2:4]),
                self.anchors,
                is_pred=False
            )

            anchor_indices = iou_anchors.argsort(
                descending=True,
                dim=0
            )

            x,y,width,height,class_label = box
            has_anchor = [False]*3

            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale

                s = self.grid_sizes[scale_idx]

                i,j = int(s*y),int(s*x)

                anchor_taken = targets[
                    scale_idx
                ][anchor_on_scale,i,j,0]

                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale,i,j,0] = 1

                    x_cell = s*x - j
                    y_cell = s*y - i

                    width_cell = width*s
                    height_cell = height*s

                    box_coordinates = torch.tensor(
                        [x_cell,y_cell,width_cell,height_cell]
                    )

                    targets[scale_idx][anchor_on_scale,i,j,1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale,i,j,5] = int(class_label)
                    has_anchor[scale_idx] = True
                    
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale,i,j,0] = -1

        return image,tuple(targets)

    def is_valid_label_file(self,label_path):
        try:
            bboxes = np.loadtxt(label_path,delimiter=" ",ndmin=2)
            return bboxes.shape[1] >= 5
        except:
            return False