"""
Install script for sd-webui-ai-image-detector extension.
Runs once during first installation to ensure dependencies are available.
"""

import launch

packages = {
    "timm": "timm>=0.9.0",
    "pytorch_grad_cam": "pytorch-grad-cam>=1.4.0",
    "safetensors": "safetensors",
    "huggingface_hub": "huggingface_hub",
}

for import_name, pip_name in packages.items():
    if not launch.is_installed(import_name):
        launch.run_pip(f"install {pip_name}", f"sd-webui-ai-image-detector: {pip_name}")
