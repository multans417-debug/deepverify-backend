"""Video classifier: per-frame ResNet18 features -> GRU over time -> one video-level logit.

Aggregation: the GRU consumes the time-ordered face-crop sequence and its LAST hidden state is mapped to a
single logit (so the decision uses temporal context, not an average of independent frame decisions).
"""
from __future__ import annotations

from torch import nn

ARCH = "resnet18-gru"


class VideoDeepfakeNet(nn.Module):
    def __init__(self, hidden: int = 128, dropout: float = 0.3, pretrained: bool = False):
        super().__init__()
        from torchvision import models
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=weights)
        self.encoder = nn.Sequential(*list(net.children())[:-1])   # conv1..layer4 + global avg-pool -> (N,512,1,1)
        self.gru = nn.GRU(512, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)
        self.frozen_modules: list = []

    def freeze(self, mode: str) -> None:
        kids = list(self.encoder.children())     # 0 conv1,1 bn1,2 relu,3 maxpool,4 layer1,5 layer2,6 layer3,7 layer4,8 pool
        groups = {"none": [], "early": kids[:6], "backbone": kids[:8]}[mode]
        for m in groups:
            for p in m.parameters():
                p.requires_grad = False
        self.frozen_modules = groups

    def forward(self, x):                        # x: (B,T,3,H,W) -> logits (B,)
        b, t = x.shape[:2]
        f = self.encoder(x.reshape(b * t, *x.shape[2:])).flatten(1).reshape(b, t, -1)
        _, h = self.gru(f)
        return self.head(self.drop(h[-1])).squeeze(1)
