import os
from typing import List, Tuple
import torch

# -----------------------------------------------------
# constants
# -----------------------------------------------------

DATA_DIR: str = "data"
DATA_DIR_1: str = os.path.join(DATA_DIR, "tumor_mri")
DATA_DIR_2: str = os.path.join(DATA_DIR, "tumor_mri_2")
DATA_DIR_3: str = os.path.join(DATA_DIR, "tumor_mri_3")

# -----------------------------------------------------

TRAIN_DIR_1: str = os.path.join(DATA_DIR_1, "Training")
TEST_DIR_1: str = os.path.join(DATA_DIR_1, "Testing")

TRAIN_DIR_2: str = os.path.join(DATA_DIR_2, "Training")
TEST_DIR_2: str = os.path.join(DATA_DIR_2, "Testing")

# -----------------------------------------------------


IMG_SIZE: Tuple[int, int] = (150, 150)
BATCH_SIZE: int = 32
VALID_SPLIT: float = 0.2
EPOCHS: int = 40
CLASS_NAMES: List[str] = ["glioma", "meningioma", "notumor", "pituitary"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES: int = len(CLASS_NAMES)
OUTPUT_MODELS_DIR = os.path.join("output", "models")
OUTPUT_PLOTS_DIR = os.path.join("output", "plots")
