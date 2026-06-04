"""
AI Image Detector — Stable Diffusion WebUI Extension
Detects AI-generated images using various classification models with CAM visualization.
Compatible with A1111, Forge, and Reforge.
"""

import sys
import os
import gradio as gr

# Add the extension's lib directory to sys.path so our modules are importable
ext_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lib_dir = os.path.join(ext_dir, "lib")
if lib_dir not in sys.path:
    sys.path.insert(0, lib_dir)

from modules import script_callbacks, scripts, processing, images, shared
from PIL import Image

from lib.detector import detector
from lib.models import CKPT_META, DEFAULT_CKPT


# ----------------------------
# 1. Standalone Tab
# ----------------------------
def on_ui_tabs():
    """Register the AI Detector as a standalone top-level tab."""

    def analyze_image(image, ckpt_name, enable_viz, method, pooling_type):
        if image is None:
            return None, None
        return detector.predict(image, ckpt_name, enable_viz, method, pooling_type)

    with gr.Blocks(analytics_enabled=False) as detector_tab:
        gr.Markdown(
            "## 🕵️ AI Generated Image Detector\n"
            "Detect whether an image is AI-generated using various classification models. "
            "Supports **DINOv3**, **CaFormer**, and more. "
            "Includes **Letterbox Scaling**, **Pooling Override**, and **CAM Heatmap Visualization**."
        )

        with gr.Row(equal_height=False):
            # --- Left Column: Input & Controls ---
            with gr.Column(scale=1):
                in_img = gr.Image(
                    type="pil",
                    label="Input Image",
                    elem_id="ai_detector_input_image",
                )

                sel_ckpt = gr.Dropdown(
                    choices=list(CKPT_META.keys()),
                    value=DEFAULT_CKPT,
                    label="Model Checkpoint",
                    elem_id="ai_detector_model_select",
                )

                sel_pooling = gr.Dropdown(
                    choices=["avg", "max", "avgmax"],
                    value="avg",
                    label="Pooling Type (Override)",
                    info="avg: Global Average (Default) | max: Global Max | avgmax: Avg + Max",
                    elem_id="ai_detector_pooling_select",
                )

                with gr.Group():
                    enable_viz = gr.Checkbox(
                        label="Generate Heatmap Visualization",
                        value=True,
                        elem_id="ai_detector_enable_viz",
                    )
                    viz_method = gr.Dropdown(
                        choices=["LayerCAM", "GradCAM++", "GradCAM"],
                        value="LayerCAM",
                        label="CAM Method",
                        elem_id="ai_detector_cam_method",
                    )

                run_btn = gr.Button(
                    "🔍 Analyze",
                    variant="primary",
                    elem_id="ai_detector_analyze_btn",
                )

            # --- Right Column: Results ---
            with gr.Column(scale=1):
                out_lbl = gr.Label(
                    num_top_classes=4,
                    label="Prediction Confidence",
                    elem_id="ai_detector_output_label",
                )
                out_viz = gr.Image(
                    type="pil",
                    label="Attention Heatmap",
                    elem_id="ai_detector_output_viz",
                )

        run_btn.click(
            fn=analyze_image,
            inputs=[in_img, sel_ckpt, enable_viz, viz_method, sel_pooling],
            outputs=[out_lbl, out_viz],
        )

    return [(detector_tab, "AI Detector", "ai_detector_tab")]


script_callbacks.on_ui_tabs(on_ui_tabs)


# ----------------------------
# 2. Script Class for txt2img / img2img Integration
# ----------------------------
class AIDetectorScript(scripts.Script):
    """
    Adds an AI Detection panel to txt2img and img2img.
    When enabled, automatically checks each generated image and appends
    the detection result to the generation info.
    """

    def title(self):
        return "AI Image Detector"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("🕵️ AI Image Detector", open=False, elem_id="ai_detector_script_accordion"):
            enabled = gr.Checkbox(
                label="Enable post-generation AI detection",
                value=False,
                elem_id="ai_detector_script_enabled",
            )
            ckpt = gr.Dropdown(
                choices=list(CKPT_META.keys()),
                value=DEFAULT_CKPT,
                label="Detection Model",
                elem_id="ai_detector_script_model",
            )
            pooling = gr.Dropdown(
                choices=["avg", "max", "avgmax"],
                value="avg",
                label="Pooling Type",
                elem_id="ai_detector_script_pooling",
            )

        # Return in same order as they'll be passed to postprocess_image
        return [enabled, ckpt, pooling]

    def postprocess_image(self, p, pp, enabled, ckpt, pooling):
        """Called after each image is generated. Runs detection if enabled."""
        if not enabled:
            return

        try:
            image = pp.image
            if image is None:
                return

            # Convert to PIL if needed
            if not isinstance(image, Image.Image):
                return

            result_dict, predicted_label, confidence = detector.predict_simple(
                image, ckpt, pooling_type=pooling
            )

            # Build info string
            detection_info = f"AI Detection: {predicted_label} ({confidence:.1%})"
            detail_parts = [f"{k}: {v:.1%}" for k, v in result_dict.items()]
            detection_detail = " | ".join(detail_parts)

            # Append to generation info
            if pp.info:
                pp.info += f"\n{detection_info} [{detection_detail}]"
            else:
                pp.info = f"{detection_info} [{detection_detail}]"

            print(f"[AI Detector] {detection_info} [{detection_detail}]")

        except Exception as e:
            print(f"[AI Detector] Post-generation detection error: {e}")
