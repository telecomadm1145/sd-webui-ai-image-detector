"""
Model definitions and checkpoint metadata for AI image detection.
"""

import torch
import torch.nn as nn
import timm

# --- Constants ---
SEED = 4421
DROP_RATE = 0.1

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ----------------------------
# Checkpoint Metadata
# ----------------------------
CKPT_META = {
    "caformer_b36.old": {
        "num_classes": 4,
        "head": "v7",
        "backbone": "caformer_b36.sail_in22k_ft_in1k_384",
        "repo_id": "telecomadm1145/swin-ai-detection",
        "filename": "caformer_b36_4class_96.safetensors",
        "labels": ["non_ai", "ai", "ani_non_ai", "ani_ai"],
        "input_size": 384,
        "cam_reshape": True,
    },
    "caformer_b36.v2": {
        "num_classes": 2,
        "head": "timm_cross_entropy",
        "backbone_timm_name": "hf-hub:animetimm/caformer_b36.dbv4-full",
        "repo_id": "telecomadm1145/danbooru-real-vs-ai-caformer-b36-v2",
        "filename": "pytorch_model.bin",
        "labels": ["AI", "Non-AI"],
        "input_size": 384,
        "cam_reshape": True,
    },
    "deepghs/cls-ai-check-1m.caformer_s36.r512": {
        "num_classes": 2,
        "head": "timm_cross_entropy",
        "backbone_timm_name": "caformer_s36.sail_in22k_ft_in1k_384",
        "repo_id": "deepghs/cls-ai-check-1m.caformer_s36.r512",
        "filename": "model.safetensors",
        "labels": ["AI", "Non-AI"],
        "input_size": 512,
        "cam_reshape": True,
    },
    "dinov3_test2": {
        "num_classes": 2,
        "head": "timm_cross_entropy",
        "backbone_timm_name": "vit_base_patch16_dinov3.lvd1689m",
        "repo_id": "telecomadm1145/vit_base.dinov3_ai_chk_test",
        "filename": "pytorch_model.bin",
        "labels": ["AI", "Non-AI"],
        "input_size": 256,
        "cam_reshape": True,
    },
}

DEFAULT_CKPT = "dinov3_test2"


# ----------------------------
# Model Definitions
# ----------------------------
class TimmClassifierWithHead(nn.Module):
    """Custom classifier with a multi-layer head on top of a timm backbone."""

    def __init__(self, model_name, num_classes, pooling_type="avg", pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=pooling_type,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(DROP_RATE),
            nn.Linear(self.backbone.num_features, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(DROP_RATE * 0.8),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
