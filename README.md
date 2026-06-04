# 🕵️ sd-webui-ai-image-detector

A [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) extension for detecting AI-generated images. Compatible with **A1111**, **Forge**, and **Reforge**.

Classifies images as AI-generated or real using state-of-the-art vision models, with optional **CAM heatmap visualization** to highlight the regions the model focuses on.

## ✨ Features

- **Multiple Detection Models**
  - `dinov3_test2` — ViT-Base with DINOv3 backbone (256px, fast)
  - `caformer_b36.v2` — CaFormer-B36 fine-tuned on Danbooru (384px)
  - `caformer_b36.old` — CaFormer-B36 4-class (AI / non-AI / anime-AI / anime-non-AI)
  - `deepghs/cls-ai-check-1m.caformer_s36.r512` — CaFormer-S36 trained on 1M samples (512px)

- **CAM Heatmap Visualization**
  - LayerCAM, GradCAM++, GradCAM
  - Smart reshape for both CNN and Vision Transformer architectures
  - Automatic DINOv2/v3 register token handling

- **WebUI Integration**
  - 🔍 Standalone **"AI Detector"** tab for manual image analysis
  - ⚡ **Post-generation hook** — automatically detect after txt2img / img2img (optional)
  - Detection results appended to generation info

- **Preprocessing**
  - Letterbox (pad) scaling — no distortion
  - Pooling type override (avg / max / avgmax)

## 📦 Installation

### Method 1: From URL (Recommended)

1. Open your WebUI
2. Go to **Extensions** → **Install from URL**
3. Paste:
   ```
   https://github.com/telecomadm1145/sd-webui-ai-image-detector.git
   ```
4. Click **Install**
5. Restart WebUI

### Method 2: Manual

```bash
cd stable-diffusion-webui/extensions
git clone https://github.com/telecomadm1145/sd-webui-ai-image-detector.git
```

Restart the WebUI. Dependencies (`timm`, `pytorch-grad-cam`, `safetensors`, `huggingface_hub`) are installed automatically on first launch.

## 🚀 Usage

### Standalone Tab

1. Navigate to the **"AI Detector"** tab
2. Upload or drag an image
3. Select a model checkpoint and CAM method
4. Click **🔍 Analyze**
5. View the confidence scores and attention heatmap

### Post-Generation Detection

1. In **txt2img** or **img2img**, expand the **"🕵️ AI Image Detector"** accordion
2. Check **"Enable post-generation AI detection"**
3. Select a model
4. Generate images as usual — detection results are printed in the console and appended to generation info

## 🏗️ Supported Models

| Name | Backbone | Classes | Input Size | Source |
|------|----------|---------|------------|--------|
| `dinov3_test2` | ViT-Base (DINOv3) | AI / Non-AI | 256 | [HF](https://huggingface.co/telecomadm1145/vit_base.dinov3_ai_chk_test) |
| `caformer_b36.v2` | CaFormer-B36 | AI / Non-AI | 384 | [HF](https://huggingface.co/telecomadm1145/danbooru-real-vs-ai-caformer-b36-v2) |
| `caformer_b36.old` | CaFormer-B36 | 4-class | 384 | [HF](https://huggingface.co/telecomadm1145/swin-ai-detection) |
| `deepghs/...r512` | CaFormer-S36 | AI / Non-AI | 512 | [HF](https://huggingface.co/deepghs/cls-ai-check-1m.caformer_s36.r512) |

Models are automatically downloaded from HuggingFace on first use and cached in the `checkpoints/` directory.

## 🔧 Requirements

- Stable Diffusion WebUI (A1111 / Forge / Reforge)
- Python 3.10+
- PyTorch with CUDA (recommended) or CPU
- Dependencies (auto-installed):
  - `timm >= 0.9.0`
  - `pytorch-grad-cam >= 1.4.0`
  - `safetensors`
  - `huggingface_hub`

## 📄 License

[MIT License](LICENSE)

## 🙏 Credits

- [timm](https://github.com/huggingface/pytorch-image-models) — PyTorch Image Models
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) — CAM visualization
- Model weights by [telecomadm1145](https://huggingface.co/telecomadm1145) and [deepghs](https://huggingface.co/deepghs)
