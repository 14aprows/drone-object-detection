import torch 
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_ROOT = os.path.join(BASE_DIR, "Data", "NewDatabase1")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LOAD_MODEL = False
SAVE_MODEL = True

CHECKPOINT_FILE = "checkpoint.pth.tar"

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]

BATCH_SIZE = 8
LEARNING_RATE = 1e-5
EPOCHS = 20

IMAGE_SIZE = 416
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]

CLASSES_LABELS = ["drone"]
NUM_CLASSES = len(CLASSES_LABELS)

