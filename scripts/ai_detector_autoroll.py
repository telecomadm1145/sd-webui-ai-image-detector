"""
AI Detector Auto-Roll Script for Stable Diffusion WebUI.
Automatically re-generates images until they score below an AI detection threshold.
Appears as a selectable script in the txt2img / img2img Script dropdown.
"""

import sys
import os
import copy

import gradio as gr

# Ensure lib is importable
ext_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lib_dir = os.path.join(ext_dir, "lib")
if lib_dir not in sys.path:
    sys.path.insert(0, lib_dir)

from modules import scripts, processing
from PIL import Image

from lib.detector import detector, get_ai_score
from lib.models import CKPT_META, DEFAULT_CKPT


class AIDetectorAutoRoll(scripts.Script):
    """
    Re-rolls image generation until AI detection score drops below a threshold.
    Selectable from the Script dropdown in txt2img / img2img.
    """

    def title(self):
        return "🎲 AI Detector Auto-Roll"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        gr.Markdown(
            "**Auto-Roll**: Repeatedly generates images and keeps only those whose "
            "AI detection score is **below** the threshold. Useful for filtering out "
            "images that look too obviously AI-generated."
        )

        ckpt = gr.Dropdown(
            choices=list(CKPT_META.keys()),
            value=DEFAULT_CKPT,
            label="Detection Model",
            elem_id="ai_autoroll_model",
        )
        pooling = gr.Dropdown(
            choices=["avg", "max", "avgmax"],
            value="avg",
            label="Pooling Type",
            elem_id="ai_autoroll_pooling",
        )
        threshold = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            value=0.5,
            label="Max AI Score (reject above this)",
            info="Images with AI score above this threshold will be discarded and re-rolled",
            elem_id="ai_autoroll_threshold",
        )
        max_rolls = gr.Slider(
            minimum=1,
            maximum=100,
            step=1,
            value=10,
            label="Max Roll Attempts",
            info="Maximum number of generation attempts before giving up",
            elem_id="ai_autoroll_max_rolls",
        )
        keep_best_on_fail = gr.Checkbox(
            label="Keep best image if all rolls fail",
            value=True,
            info="If no image passes the threshold, keep the one with the lowest AI score",
            elem_id="ai_autoroll_keep_best",
        )

        return [ckpt, pooling, threshold, max_rolls, keep_best_on_fail]

    def run(self, p, ckpt, pooling, threshold, max_rolls, keep_best_on_fail):
        """
        Main auto-roll loop.
        Generates images, checks AI score, keeps good ones, re-rolls bad ones.
        """
        max_rolls = int(max_rolls)
        desired_count = p.n_iter * p.batch_size

        # Collect results
        good_images = []        # (image, infotext, seed, ai_score)
        best_rejected = []      # Track best rejected for fallback

        # Pre-load the detection model
        print(f"[AI Auto-Roll] Starting with threshold={threshold:.0%}, max_rolls={max_rolls}")
        print(f"[AI Auto-Roll] Need {desired_count} images, generating up to {max_rolls} rounds")

        initial_seed = p.seed if p.seed != -1 else None
        roll_count = 0

        while len(good_images) < desired_count and roll_count < max_rolls:
            roll_count += 1

            # Create a copy of processing params for this round
            p2 = copy.copy(p)
            p2.n_iter = 1
            p2.batch_size = min(p.batch_size, desired_count - len(good_images))

            # Vary seed each round (keep first round's seed as-is)
            if roll_count > 1 and initial_seed is not None:
                p2.seed = initial_seed + roll_count - 1
            elif roll_count > 1:
                p2.seed = -1  # Random seed each roll

            p2.do_not_save_grid = True

            print(f"[AI Auto-Roll] Round {roll_count}/{max_rolls} (seed={p2.seed}, need {desired_count - len(good_images)} more)...")

            processed = processing.process_images(p2)

            # Check each generated image
            for i, img in enumerate(processed.images):
                if not isinstance(img, Image.Image):
                    continue

                # Skip grid images (first image when batch_size > 1 and grid is enabled)
                # Grid images are typically larger than individual images
                if i == 0 and len(processed.images) > p2.batch_size:
                    continue

                result_dict, predicted_label, confidence = detector.predict_simple(
                    img, ckpt, pooling_type=pooling
                )
                ai_score = get_ai_score(result_dict)

                # Build infotext
                info_idx = min(i, len(processed.infotexts) - 1)
                infotext = processed.infotexts[info_idx] if processed.infotexts else ""
                seed_val = processed.all_seeds[min(i, len(processed.all_seeds) - 1)] if processed.all_seeds else p2.seed

                score_info = f"AI Score: {ai_score:.1%} (threshold: {threshold:.0%})"

                if ai_score <= threshold:
                    # ✅ Passed!
                    print(f"[AI Auto-Roll]   ✅ Image passed — {score_info}")
                    full_info = f"{infotext}\n{score_info} [PASS, roll #{roll_count}]"
                    good_images.append((img, full_info, seed_val, ai_score))

                    if len(good_images) >= desired_count:
                        break
                else:
                    # ❌ Rejected
                    print(f"[AI Auto-Roll]   ❌ Image rejected — {score_info}")
                    full_info = f"{infotext}\n{score_info} [REJECTED, roll #{roll_count}]"
                    best_rejected.append((img, full_info, seed_val, ai_score))

        # --- Assemble final results ---
        if len(good_images) < desired_count and keep_best_on_fail and best_rejected:
            # Sort rejected by AI score (ascending = best first)
            best_rejected.sort(key=lambda x: x[3])
            shortage = desired_count - len(good_images)
            fill = best_rejected[:shortage]
            for img, info, seed, score in fill:
                patched_info = info.replace("[REJECTED,", "[BEST-EFFORT,")
                good_images.append((img, patched_info, seed, score))
                print(f"[AI Auto-Roll]   🔄 Filling with best rejected (AI score: {score:.1%})")

        # Build final Processed object
        final_images = [item[0] for item in good_images]
        final_infotexts = [item[1] for item in good_images]
        final_seeds = [item[2] for item in good_images]

        # Create a result based on the last processed object
        result = processing.Processed(
            p,
            images_list=final_images,
            seed=final_seeds[0] if final_seeds else p.seed,
            info=final_infotexts[0] if final_infotexts else "",
            infotexts=final_infotexts,
        )

        passed = sum(1 for _, _, _, s in good_images if s <= threshold)
        filled = len(good_images) - passed

        summary = (
            f"\n\n--- AI Auto-Roll Summary ---\n"
            f"Rounds: {roll_count}/{max_rolls}\n"
            f"Passed: {passed}/{desired_count}\n"
        )
        if filled > 0:
            summary += f"Best-effort fills: {filled}\n"
        summary += f"Threshold: {threshold:.0%}\n"

        print(summary)

        return result
