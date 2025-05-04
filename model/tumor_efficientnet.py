from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from common import DEVICE, IMG_SIZE, NUM_CLASSES


class TumorEfficientNet(nn.Module):
    """
    EfficientNet-B0 adapted for tumor classification.
    - Loads a pretrained backbone (optional).
    - Replaces the default head with a two-layer classifier.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.backbone = self._create_backbone(pretrained)
        feat_dim = self._infer_feature_dim()
        self.classifier = self._create_classifier(feat_dim, num_classes, dropout_rate)
        self._init_weights()

    @staticmethod
    def _create_backbone(pretrained: bool) -> nn.Module:
        """
        Returns EfficientNet-B0 without its default classifier.
        """
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b0(weights=weights, progress=True)
        model.classifier = nn.Identity()  # strip off ImageNet head
        return model

    def _infer_feature_dim(self) -> int:
        """
        Runs a dummy forward pass to determine the output dimension
        of the backbone’s feature extractor.
        """
        # assume input size is square
        dummy = torch.zeros(1, 3, IMG_SIZE[0], IMG_SIZE[1])
        with torch.no_grad():
            feat = self.backbone(dummy)
        return feat.shape[1]

    @staticmethod
    def _create_classifier(
        in_features: int,
        num_classes: int,
        dropout_rate: float
    ) -> nn.Sequential:
        """
        Builds a simple dropout → linear head.
        """
        return nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features // 2, num_classes)
        )

    def _init_weights(self) -> None:
        """
        Applies Kaiming initialization to all Linear layers.
        """
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: backbone → classifier.
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


def build_model(
    num_classes: int,
    img_size: Tuple[int, int],
    device: torch.device,
    pretrained: bool = True,
) -> nn.Module:
    """
    Instantiates TumorEfficientNet and moves it to the target device.
    """
    model = TumorEfficientNet(num_classes=num_classes, pretrained=pretrained)
    return model.to(device)


def build_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4
) -> optim.Optimizer:
    """
    Constructs an Adam optimizer for the model parameters.
    """
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


if __name__ == "__main__":
    # Quick sanity check
    model = build_model(NUM_CLASSES, IMG_SIZE, DEVICE, pretrained=True)
    optimizer = build_optimizer(model)
    print(model)
    print(optimizer)
