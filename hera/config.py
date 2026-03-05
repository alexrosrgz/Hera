import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Video settings
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
CAPTURE_FPS = 30

# Model paths
MODELS_DIR = os.path.join(ROOT_DIR, "models")
SWAPPER_MODEL_PATH = os.path.join(MODELS_DIR, "inswapper_128_fp16.onnx")
ENHANCER_MODEL_PATH = os.path.join(MODELS_DIR, "GFPGANv1.4.onnx")

# Reference face
REFERENCES_DIR = os.path.join(ROOT_DIR, "references")
DEFAULT_REFERENCE_PATH = os.path.join(REFERENCES_DIR, "influencer.png")

# Recordings
RECORDINGS_DIR = os.path.join(ROOT_DIR, "recordings")

# ONNX Runtime execution providers (CoreML for M1 GPU acceleration)
EXECUTION_PROVIDERS = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

# Face detection
FACE_ANALYSER_MODEL = "buffalo_l"
FACE_DETECTION_INTERVAL = 3  # Re-detect every N frames

# Enhancement
ENHANCE_ENABLED = True
