"""MS-EffGCViT Image Classifier: Multi-Scale Efficient Global Context Vision Transformer.
Pretrained on KoDF / Celeb-DF, outputs a single pre-sigmoid logit for class FAKE.
"""
from __future__ import annotations

import torch
from torch import nn

ARCH = "ms_eff_gcvit_b0"


class GCViTDeepfakeNet(nn.Module):
    def __init__(self, variant: str = "b0", dataset: str = "kodf"):
        super().__init__()
        from deepguard import ms_eff_gcvit_b0
        self.net = ms_eff_gcvit_b0(pretrained=False, dataset=dataset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)

        
        if out.ndim > 1:
            out = out.squeeze(1)
        return out
