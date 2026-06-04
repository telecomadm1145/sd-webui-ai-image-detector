"""
Utility functions for image preprocessing and CAM visualization.
"""

import numpy as np
from PIL import Image


def reshape_transform(tensor):
    """
    Smart reshape for CAM compatibility with both CNN (4D) and Transformer (3D) outputs.
    Automatically handles DINOv2/v3 register tokens.
    """
    # Case 1: 4D Tensor (CNNs or already reshaped Transformers)
    if tensor.ndim == 4:
        h, w, c = tensor.shape[1], tensor.shape[2], tensor.shape[3]
        if c < h and c < w:
            # Already (B, C, H, W)
            return tensor
        # (B, H, W, C) -> (B, C, H, W)
        return tensor.permute(0, 3, 1, 2)

    # Case 2: 3D Tensor (Vision Transformers) -> (B, Tokens, Embed_Dim)
    if tensor.ndim == 3:
        b, num_tokens, c = tensor.shape

        # Try common register token counts: 0, 1, 4, 5, 8
        for extra_tokens in [0, 1, 5, 4, 8]:
            patches = num_tokens - extra_tokens
            if patches <= 0:
                continue

            side = int(np.sqrt(patches))
            if side * side == patches:
                tensor = tensor[:, -patches:, :]
                result = tensor.reshape(b, side, side, c).permute(0, 3, 1, 2)
                return result

        # Fallback: assume 1 CLS token
        if num_tokens > 0:
            patches = num_tokens - 1
            side = int(np.sqrt(patches))
            if side * side == patches:
                tensor = tensor[:, 1:, :]
                return tensor.reshape(b, side, side, c).permute(0, 3, 1, 2)

    return tensor


def resize_with_pad(image: Image.Image, target_size: int, fill_color=(255, 255, 255)):
    """
    Resize image with letterbox padding to target_size x target_size.
    Returns the padded image and the valid region box (paste_x, paste_y, new_w, new_h).
    """
    w, h = image.size
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)

    image_resized = image.resize((new_w, new_h), Image.BICUBIC)

    new_img = Image.new("RGB", (target_size, target_size), fill_color)
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    new_img.paste(image_resized, (paste_x, paste_y))

    return new_img, (paste_x, paste_y, new_w, new_h)
