import torch
from torch.utils.data import DataLoader

from dataset.dataset import Dataset
from configs.config import *

def get_train_loader(transform=None):
    train_dataset = Dataset(
        data_dir=DATA_ROOT,
        anchors=ANCHORS,
        image_size=IMAGE_SIZE,
        grid_sizes=S,
        num_classes=NUM_CLASSES,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    return train_loader

def get_val_loader(transform=None):
    val_dataset = Dataset(
        data_dir=DATA_ROOT + "val/",
        anchors=ANCHORS,
        image_size=IMAGE_SIZE,
        grid_sizes=S,
        num_classes=NUM_CLASSES,
        transform=transform
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return val_loader