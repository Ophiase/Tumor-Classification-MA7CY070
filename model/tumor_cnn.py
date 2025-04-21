import torch
from typing import Tuple
from torch import nn
from common import DEVICE, IMG_SIZE, NUM_CLASSES


class TumorCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, img_size: Tuple[int, int] = IMG_SIZE):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (img_size[0]//8) * (img_size[1]//8), 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


if __name__ == "__main__":
    # Example usage
    model = TumorCNN(NUM_CLASSES).to(DEVICE)
    print(model)
