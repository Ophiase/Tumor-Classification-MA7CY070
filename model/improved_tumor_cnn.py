from typing import Tuple, Callable
import torch
import torch.nn as nn
import torch.optim as optim

from common import DEVICE, IMG_SIZE, NUM_CLASSES


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )


class ImprovedTumorCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        img_size: Tuple[int, int]
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def build_model(
    num_classes: int,
    img_size: Tuple[int, int],
    device: torch.device
) -> nn.Module:
    model = ImprovedTumorCNN(num_classes, img_size)
    model.apply(init_weights)
    return model.to(device)


def build_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4
) -> optim.Optimizer:
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


if __name__ == "__main__":
    # usage
    model = build_model(NUM_CLASSES, IMG_SIZE, DEVICE)
    optimizer = build_optimizer(model)
