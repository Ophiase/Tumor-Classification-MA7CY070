import os
from typing import List, Tuple
import torch


# constants
DATA_DIR: str = "data/tumor_mri"
TRAIN_DIR: str = os.path.join(DATA_DIR, "Training")
TEST_DIR: str = os.path.join(DATA_DIR, "Testing")
IMG_SIZE: Tuple[int, int] = (150, 150)
BATCH_SIZE: int = 32
VALID_SPLIT: float = 0.2
EPOCHS: int = 40
CLASS_NAMES: List[str] = ["glioma", "meningioma", "notumor", "pituitary"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES: int = len(CLASS_NAMES)
OUTPUT_MODELS_DIR = os.path.join("output", "models")
OUTPUT_PLOTS_DIR = os.path.join("output", "plots")
