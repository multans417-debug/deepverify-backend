"""Image classifier: ResNet18 backbone, single logit output. sigmoid(logit) = score for class FAKE."""
from __future__ import annotations

from torch import nn

ARCH = "resnet18-binary"


class ImageDeepfakeNet(nn.Module):
    def __init__(self, pretrained: bool = False, dropout: float = 0.3):
        super().__init__()
        from torchvision import models
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None  # ImageNet init only, NOT a detector
        self.net = models.resnet18(weights=weights)
        self.net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.net.fc.in_features, 1))
        self.frozen_modules: list = []

    def freeze(self, mode: str) -> None:
        """none: fine-tune everything | early: freeze stem+layer1+layer2 | backbone: train only the head."""
        n = self.net
        groups = {"none": [], "early": [n.conv1, n.bn1, n.layer1, n.layer2],
                  "backbone": [n.conv1, n.bn1, n.layer1, n.layer2, n.layer3, n.layer4]}[mode]
        for m in groups:
            for p in m.parameters():
                p.requires_grad = False
        self.frozen_modules = groups

    def forward(self, x):                       # x: (B,3,H,W) -> logits (B,)
        return self.net(x).squeeze(1)
