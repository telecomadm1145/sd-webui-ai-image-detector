"""
Core AI Image Detector class.
Handles model loading, preprocessing, inference, and CAM visualization.
"""

import os
import gc

import cv2
import torch
import timm
import numpy as np
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from .models import CKPT_META, TimmClassifierWithHead, IMAGENET_MEAN, IMAGENET_STD, SEED
from .utils import reshape_transform, resize_with_pad

# Seed initialization
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def _get_device():
    """Get the best available device. Uses WebUI shared device if available."""
    try:
        from modules import devices
        return devices.device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_checkpoint_dir():
    """Get the checkpoint directory relative to the extension root."""
    ext_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = os.path.join(ext_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    return ckpt_dir


class AIImageDetector:
    """
    Loads and manages AI image detection models.
    Supports multiple backbones and CAM visualization methods.
    """

    def __init__(self):
        self.model = None
        self.current_ckpt_name = None
        self.current_meta = None
        self.current_pooling_type = None

    def _get_target_layers(self, model):
        """Automatically find the best target layer for CAM visualization."""
        backbone = getattr(model, "backbone", model)
        if hasattr(backbone, "stages"):
            last_stage = backbone.stages[-1]
            if hasattr(last_stage, "blocks"):
                return [last_stage.blocks[-1]]
            return [last_stage]
        if hasattr(backbone, "layers"):
            return [backbone.layers[-1].blocks[-1]]
        if hasattr(backbone, "blocks"):
            return [backbone.blocks[-1]]
        return [list(backbone.children())[-1]]

    def load_model(self, ckpt_name, pooling_type="avg"):
        """Load a model checkpoint. Skips if already loaded with same config."""
        if (
            ckpt_name == self.current_ckpt_name
            and pooling_type == self.current_pooling_type
            and self.model is not None
        ):
            return

        device = _get_device()
        print(f"[AI Detector] Loading model: {ckpt_name} with pooling: {pooling_type}...")
        meta = CKPT_META[ckpt_name]

        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
            gc.collect()

        ckpt_dir = _get_checkpoint_dir()
        ckpt_file = hf_hub_download(
            repo_id=meta["repo_id"],
            filename=meta["filename"],
            local_dir=ckpt_dir,
            force_download=False,
        )

        head_type = meta.get("head")

        if head_type == "timm_cross_entropy":
            model = timm.create_model(
                meta["backbone_timm_name"],
                pretrained=False,
                num_classes=meta["num_classes"],
                global_pool=pooling_type,
            )
        else:
            model = TimmClassifierWithHead(
                meta["backbone"],
                num_classes=meta["num_classes"],
                pooling_type=pooling_type,
                pretrained=False,
            )

        model = model.to(device)

        if meta["filename"].endswith(".safetensors"):
            state_dict = load_file(ckpt_file, device=str(device))
        else:
            state_dict = torch.load(ckpt_file, map_location=device)

        if isinstance(state_dict, dict):
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            state_dict = {
                k.replace("module.", "").replace("model.", ""): v
                for k, v in state_dict.items()
            }

        model.load_state_dict(state_dict, strict=False)
        model.eval()

        self.model = model
        self.current_ckpt_name = ckpt_name
        self.current_meta = meta
        self.current_pooling_type = pooling_type
        print(f"[AI Detector] Model loaded successfully.")

    def preprocess(self, image: Image.Image):
        """Preprocess a PIL image for model input. Returns (tensor, padded_image, valid_box)."""
        device = _get_device()

        if image.mode != "RGB":
            if image.mode == "RGBA":
                bg = Image.new("RGBA", image.size, (255, 255, 255))
                image = bg.alpha_composite(image).convert("RGB")
            else:
                image = image.convert("RGB")

        meta = self.current_meta
        size = int(meta.get("input_size", 384))

        img_padded, valid_box = resize_with_pad(image, size, fill_color=(255, 255, 255))

        img_np = np.array(img_padded).astype(np.float32) / 255.0

        mean = np.array(meta.get("mean", IMAGENET_MEAN), dtype=np.float32).reshape(1, 1, 3)
        std = np.array(meta.get("std", IMAGENET_STD), dtype=np.float32).reshape(1, 1, 3)
        img_norm = (img_np - mean) / std

        img_tensor = (
            torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(device)
        )

        return img_tensor, img_padded, valid_box

    def predict(self, image, ckpt_name, enable_viz=True, method_name="LayerCAM", pooling_type="avg"):
        """
        Run prediction on an image.

        Args:
            image: PIL Image
            ckpt_name: Key from CKPT_META
            enable_viz: Whether to generate CAM heatmap
            method_name: CAM method name ("LayerCAM", "GradCAM++", "GradCAM")
            pooling_type: Global pooling type ("avg", "max", "avgmax")

        Returns:
            (result_dict, viz_image) — result_dict maps label->probability,
            viz_image is a PIL Image with heatmap overlay (or None).
        """
        if image is None:
            return None, None

        self.load_model(ckpt_name, pooling_type)

        input_tensor, img_padded, valid_box = self.preprocess(image)
        labels = self.current_meta["labels"]

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()

        result_dict = {labels[i]: float(probs[0][i].item()) for i in range(len(labels))}

        viz_image = None
        if enable_viz:
            try:
                target_layers = self._get_target_layers(self.model)

                cam_methods = {
                    "LayerCAM": LayerCAM,
                    "GradCAM++": GradCAMPlusPlus,
                    "GradCAM": GradCAM,
                }
                cam_cls = cam_methods.get(method_name, LayerCAM)

                cam = cam_cls(
                    model=self.model,
                    target_layers=target_layers,
                    reshape_transform=reshape_transform,
                )

                targets = [ClassifierOutputTarget(predicted_class)]
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                grayscale_cam = grayscale_cam[0, :]

                if grayscale_cam is None or np.isnan(grayscale_cam).any():
                    print("[AI Detector] Warning: CAM calculation returned NaNs or None.")
                    viz_image = image
                else:
                    x, y, w, h = valid_box
                    cam_cropped = grayscale_cam[y : y + h, x : x + w]
                    orig_w, orig_h = image.size
                    cam_resized = cv2.resize(cam_cropped, (orig_w, orig_h))
                    orig_img_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
                    visualization = show_cam_on_image(orig_img_np, cam_resized, use_rgb=True)
                    viz_image = Image.fromarray(visualization)

            except Exception as e:
                print(f"[AI Detector] Visualization Error: {e}")
                import traceback
                traceback.print_exc()
                viz_image = image

        return result_dict, viz_image

    def predict_simple(self, image, ckpt_name, pooling_type="avg"):
        """
        Lightweight prediction without visualization. Returns (result_dict, predicted_label, confidence).
        Used by the post-generation hook.
        """
        if image is None:
            return None, None, 0.0

        self.load_model(ckpt_name, pooling_type)

        input_tensor, _, _ = self.preprocess(image)
        labels = self.current_meta["labels"]

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = float(probs[0][predicted_class].item())

        result_dict = {labels[i]: float(probs[0][i].item()) for i in range(len(labels))}
        predicted_label = labels[predicted_class]

        return result_dict, predicted_label, confidence


# Global singleton instance
detector = AIImageDetector()
