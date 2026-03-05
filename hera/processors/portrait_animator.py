import logging
import os
import sys

from hera import config

logger = logging.getLogger("hera.portrait_animator")

# Enable MPS fallback for unsupported ops before any torch import
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


class PortraitAnimator:
    def __init__(self, model_dir=None, device=None):
        self.model_dir = model_dir or config.LIVEPORTRAIT_MODELS_DIR
        self.device = device or config.LIVEPORTRAIT_DEVICE
        self.pipeline = None

    def load(self, progress_callback=None):
        if progress_callback:
            progress_callback("Loading LivePortrait models...")

        lp_dir = config.LIVEPORTRAIT_DIR
        if lp_dir not in sys.path:
            sys.path.insert(0, lp_dir)

        try:
            from src.config.inference_config import InferenceConfig
            from src.config.crop_config import CropConfig
            from src.live_portrait_pipeline import LivePortraitPipeline
        except ImportError as e:
            raise RuntimeError(
                f"LivePortrait not found at {lp_dir}. "
                f"Run: git submodule add https://github.com/KwaiVGI/LivePortrait vendor/LivePortrait\n"
                f"Error: {e}"
            )

        pretrained = config.LIVEPORTRAIT_PRETRAINED_WEIGHTS
        base = os.path.join(pretrained, "liveportrait", "base_models")
        retarget = os.path.join(pretrained, "liveportrait", "retargeting_models")

        inference_cfg = InferenceConfig(
            checkpoint_F=os.path.join(base, "appearance_feature_extractor.pth"),
            checkpoint_M=os.path.join(base, "motion_extractor.pth"),
            checkpoint_G=os.path.join(base, "spade_generator.pth"),
            checkpoint_W=os.path.join(base, "warping_module.pth"),
            checkpoint_S=os.path.join(retarget, "stitching_retargeting_module.pth"),
            flag_force_cpu=(self.device == "cpu"),
        )

        crop_cfg = CropConfig(
            landmark_ckpt_path=os.path.join(pretrained, "liveportrait", "landmark.onnx"),
            insightface_root=os.path.join(pretrained, "insightface"),
            flag_force_cpu=(self.device == "cpu"),
        )

        self.pipeline = LivePortraitPipeline(
            inference_cfg=inference_cfg,
            crop_cfg=crop_cfg,
        )

        if progress_callback:
            progress_callback("LivePortrait models loaded.")

    def animate(self, source_image_path, driving_video_path, output_dir=None, progress_callback=None):
        if self.pipeline is None:
            raise RuntimeError("Models not loaded. Call load() first.")

        if not os.path.exists(source_image_path):
            raise FileNotFoundError(f"Source image not found: {source_image_path}")
        if not os.path.exists(driving_video_path):
            raise FileNotFoundError(f"Driving video not found: {driving_video_path}")

        output_dir = output_dir or config.GENERATE_OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)

        if progress_callback:
            progress_callback("generating", None)

        lp_dir = config.LIVEPORTRAIT_DIR
        if lp_dir not in sys.path:
            sys.path.insert(0, lp_dir)

        from src.config.argument_config import ArgumentConfig

        args = ArgumentConfig()
        args.source = source_image_path
        args.driving = driving_video_path
        args.output_dir = output_dir

        try:
            wfp, wfp_concat = self.pipeline.execute(args)
            return wfp
        except Exception as e:
            logger.error(f"Animation failed: {e}")
            raise

    def animate_frames(self, source_image_path, driving_video_path, output_dir=None, progress_callback=None, cancel_event=None):
        """Animation with indeterminate progress (LivePortrait doesn't expose per-frame callbacks)."""
        if progress_callback:
            progress_callback("generating", None)

        result = self.animate(
            source_image_path,
            driving_video_path,
            output_dir=output_dir,
            progress_callback=progress_callback,
        )

        if progress_callback:
            progress_callback("done", None)

        return result
