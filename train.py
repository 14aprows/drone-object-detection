import torch
import torch.optim as optim

import configs.config as config

from model.YOLOV3.yolov3 import YOLOv3
from loss.yolo_loss import YOLOLoss
from dataset.dataloader import get_train_loader
from transforms.transforms import get_train_transform
from engine.trainer import Trainer


def main():
    device = config.DEVICE

    model = YOLOv3(
        num_classes=config.NUM_CLASSES
    ).to(device)


    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE
    )

    loss_fn = YOLOLoss()

    train_loader = get_train_loader(
        transform=get_train_transform(config.IMAGE_SIZE)
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        anchors=config.ANCHORS,
        device=device,
        config=config
    )

    trainer.fit()

if __name__ == "__main__":
    main()