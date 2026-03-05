# Hera

Real-time AI face swap desktop app for macOS. Captures webcam input, swaps your face with an AI-generated influencer face using ML models, and records the output as video with audio.

Built for Apple Silicon (M1/M2/M3) with CoreML acceleration.

## How It Works

```
Webcam (OpenCV) -> Face Detection (InsightFace) -> Face Swap (InSwapper) -> Face Enhance (GFPGAN) -> Preview + Record
```

## Features

- **Real-time face swap** using InsightFace + InSwapper ONNX model
- **Face enhancement** with GFPGAN for sharper, more realistic output
- **Live preview** with FPS counter
- **Video recording** with audio capture and FFmpeg muxing
- **CoreML acceleration** for native M1/M2/M3 GPU performance
- **Optimized pipeline** — threaded architecture with frame dropping and cached face detection

## Requirements

- macOS with Apple Silicon (M1 Pro+ recommended)
- Python 3.10
- FFmpeg (for audio muxing)

## Setup

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download ML models (~600MB)
python setup_models.py
```

## Usage

```bash
# Launch the app
python main.py

# Options
python main.py --reference path/to/face.png   # Set reference face
python main.py --no-enhance                    # Disable GFPGAN for higher FPS
python main.py --detection-interval 5          # Detect face every 5 frames
python main.py --resolution 1280x720           # Set capture resolution
```

### Getting Started

1. Generate an AI influencer face using [Draw Things](https://apps.apple.com/app/draw-things-ai-generation/id6444050820) or similar
2. Save it to `references/influencer.png`
3. Run `python main.py`
4. Click **Select Reference Face** if not auto-loaded
5. Hit **Start Recording** to capture video

## Tech Stack

| Component | Tool |
|---|---|
| Face detection | InsightFace (`buffalo_l`) |
| Face swap | `inswapper_128_fp16.onnx` |
| Face enhancement | `GFPGANv1.4.onnx` |
| Inference | ONNX Runtime + CoreML EP |
| Webcam / Recording | OpenCV |
| UI | CustomTkinter |

## Project Structure

```
Hera/
├── main.py                  # Entry point
├── setup_models.py          # Model downloader
├── requirements.txt
├── models/                  # ONNX models (gitignored)
├── references/              # Reference face images
├── recordings/              # Output videos (gitignored)
└── hera/
    ├── config.py            # Global config
    ├── capture.py           # Webcam capture
    ├── face_analyser.py     # Face detection with caching
    ├── pipeline.py          # Threaded processing pipeline
    ├── recorder.py          # Video + audio recording
    ├── processors/
    │   ├── face_swapper.py  # InSwapper inference
    │   └── face_enhancer.py # GFPGAN enhancement
    └── ui/
        └── app.py           # Desktop GUI
```

## Acknowledgments

- [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam) — architecture reference
- [InsightFace](https://github.com/deepinsight/insightface) — face analysis + swap models
- [GFPGAN](https://github.com/TencentARC/GFPGAN) — face enhancement
