# Hera - Real-Time AI Influencer Face Swap App

## Context

Build a desktop app that lets a user (male) record video content where their face is replaced with an AI-generated female influencer face in real-time. The app captures webcam input, swaps the face using ML models, and records the output as video files for later posting to social media. Target hardware: M1 Pro MacBook with 16GB RAM.

Inspired by content like [@ainterestingupdate](https://www.instagram.com/ainterestingupdate/) on Instagram.

## Architecture Overview

```
Webcam (OpenCV) -> Face Detection (InsightFace/SCRFD) -> Face Swap (InSwapper) -> Face Enhance (GFPGAN) -> Preview + Record
```

## Tech Stack

| Component | Tool | Why |
|---|---|---|
| Language | Python 3.10 | Best CoreML + insightface compatibility |
| Face detection | InsightFace (`buffalo_l`) | Industry standard, ONNX-based |
| Face swap | `inswapper_128_fp16.onnx` | Proven model, 265MB, CoreML compatible |
| Face enhancement | `GFPGANv1.4.onnx` | ONNX version avoids PyTorch dependency |
| Inference runtime | `onnxruntime` with CoreML EP | Native M1 GPU acceleration |
| Webcam capture | OpenCV | Standard, works on macOS |
| Video recording | OpenCV `VideoWriter` | Record processed frames to MP4 |
| UI | CustomTkinter | Native desktop feel, proven by Deep-Live-Cam |
| AI face generation | Draw Things (Mac app) | Free, one-time task, not part of codebase |

## Project Structure

```
Hera/
├── main.py                          # Entry point, arg parsing
├── requirements.txt                 # Python dependencies
├── models/                          # ONNX models (gitignored)
│   ├── inswapper_128_fp16.onnx
│   └── GFPGANv1.4.onnx
├── references/                      # AI influencer reference face images
├── recordings/                      # Output video recordings
├── hera/
│   ├── __init__.py
│   ├── config.py                    # Global config/state (resolution, FPS, model paths)
│   ├── capture.py                   # Webcam capture (OpenCV)
│   ├── face_analyser.py             # InsightFace wrapper: detect, get landmarks, cache
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── face_swapper.py          # InSwapper inference + color correction
│   │   └── face_enhancer.py         # GFPGAN ONNX inference
│   ├── recorder.py                  # Video recording (OpenCV VideoWriter)
│   ├── pipeline.py                  # Orchestrates capture -> process -> preview/record
│   └── ui/
│       ├── __init__.py
│       └── app.py                   # CustomTkinter GUI
```

## Implementation Steps

### Step 1: Project Setup
- Create project structure and virtual environment (Python 3.10)
- Install core dependencies: `insightface`, `onnxruntime`, `opencv-python`, `numpy`, `Pillow`, `customtkinter`
- Create `requirements.txt`
- Create `config.py` with defaults (640x480, 30fps, model paths, CoreML provider)

### Step 2: Webcam Capture Module (`capture.py`)
- OpenCV `VideoCapture(0)` with configurable resolution/FPS
- Frame iterator/callback interface for the pipeline
- Graceful camera release on shutdown
- macOS AVFoundation backend (automatic)

### Step 3: Face Analysis Module (`face_analyser.py`)
- Initialize InsightFace `FaceAnalysis` with `buffalo_l` model
- `detect_faces(frame)` -> list of face objects with bounding boxes + landmarks
- `get_one_face(frame)` -> single face (largest or highest confidence)
- **Key optimization**: Cache face detection results, only re-detect every 3-5 frames. Use cached bounding boxes for intermediate frames. This is critical for hitting ~10+ FPS on M1 Pro.

### Step 4: Face Swap Processor (`face_swapper.py`)
- Load `inswapper_128_fp16.onnx` via `insightface.model_zoo.get_model()`
- `swap_face(frame, source_face, target_face)` -> swapped frame
- Color correction using LAB color space matching (match skin tone of swapped face to original lighting)
- Source face = AI influencer reference image (loaded once at startup)
- Target face = detected face in webcam frame

### Step 5: Face Enhancement Processor (`face_enhancer.py`)
- Load `GFPGANv1.4.onnx` via ONNX Runtime with CoreML EP
- `enhance_face(frame, face_bbox)` -> enhanced frame
- Crop face region, run through GFPGAN, paste back
- This upscales the 128x128 swap output to look sharp and realistic
- Make this toggleable (skip for higher FPS when previewing)

### Step 6: Processing Pipeline (`pipeline.py`)
- Threaded architecture with 3 workers:
  - **Thread 1 (Capture)**: Reads webcam frames, puts into input queue
  - **Thread 2 (Detect + Swap)**: Takes frames, runs face detection (every N frames) + swap
  - **Thread 3 (Enhance)**: Takes swapped frames, runs GFPGAN enhancement
- Queue-based communication between threads (non-blocking, drop stale frames)
- Frame timestamping to discard old frames and maintain real-time feel
- Expected performance: ~10-15 FPS with enhancement, ~15-20 FPS without

### Step 7: Video Recorder (`recorder.py`)
- OpenCV `VideoWriter` with H.264 codec (`cv2.VideoWriter_fourcc(*'avc1')`)
- Start/stop recording controls
- Save to `recordings/` with timestamp filename
- Configurable output resolution (match input or upscale)
- Audio recording via `sounddevice` or `pyaudio` in parallel, mux with FFmpeg at the end

### Step 8: Desktop UI (`ui/app.py`)
- CustomTkinter window with:
  - **Live preview panel**: Shows processed webcam feed in real-time
  - **Reference face selector**: Load/change the AI influencer face image
  - **Record button**: Start/stop recording with visual indicator
  - **Settings panel**:
    - Toggle face enhancement on/off (FPS vs quality tradeoff)
    - Face detection frequency slider (every 1-5 frames)
    - Output resolution selector
  - **Status bar**: Current FPS, recording duration, model status
- Preview runs via `after()` callback loop pulling from the output queue

### Step 9: Audio Support
- Capture microphone audio alongside video using `sounddevice`
- Save audio as WAV file during recording
- On stop: mux video + audio with FFmpeg (`ffmpeg -i video.mp4 -i audio.wav -c:v copy -c:a aac output.mp4`)
- Optional future: real-time voice conversion with RVC

### Step 10: Polish and Optimization
- Model preloading with startup progress indicator
- Graceful error handling (camera not found, model missing)
- Memory optimization for 16GB constraint (release models when not in use)
- Frame interpolation if FPS drops below threshold

## Model Download Setup

Models are too large for git. On first run or via setup script:

1. **buffalo_l** (face analysis): Auto-downloads to `~/.insightface/models/buffalo_l/` on first `FaceAnalysis` init
2. **inswapper_128_fp16.onnx** (~265MB): Download from InsightFace community mirrors -> `models/`
3. **GFPGANv1.4.onnx** (~350MB): Download from HuggingFace (Gourieff/ReActor) -> `models/`

Create a `setup_models.py` script that downloads these automatically.

## Key Optimizations for 16GB M1 Pro

1. **ONNX-only pipeline** (no PyTorch) - saves ~2-4GB RAM
2. **CoreML execution provider** - uses Neural Engine + GPU natively
3. **FP16 swap model** - half the size of FP32 (265MB vs 530MB)
4. **Cached face detection** - run detection every 3-5 frames, reuse bounding boxes
5. **Frame dropping** - if processing is slower than capture, drop old frames instead of queuing
6. **Single face mode** - only detect/swap one face (skip multi-face overhead)

## Generating Your AI Influencer Face

This is a one-time task, separate from the app:

1. Install **Draw Things** from the Mac App Store (free)
2. Use a prompt like: "professional headshot of a young woman, photorealistic, front-facing, neutral background, 8k"
3. Generate several candidates, pick the best one
4. Save to `references/influencer.png`

## Verification Plan

1. **Step 1-2**: Run `main.py`, verify webcam opens and displays raw frames in an OpenCV window
2. **Step 3**: Load a test image, verify face detection returns bounding box + landmarks
3. **Step 4**: Load reference face + test frame, verify face swap produces output (save to file)
4. **Step 5**: Verify GFPGAN enhancement improves swap quality (compare before/after)
5. **Step 6**: Run full pipeline, verify live preview shows swapped face at 10+ FPS
6. **Step 7**: Record a 10-second clip, verify MP4 plays correctly
7. **Step 8**: Launch UI, verify all controls work (reference face, record, settings)
8. **Step 9**: Record with audio, verify final MP4 has synced video + audio
9. **End-to-end**: Record a 30-second video acting as an AI influencer, review quality

## Reference Projects

- [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam) - Main architecture reference
- [InsightFace](https://github.com/deepinsight/insightface) - Face analysis + swap models
- [GFPGAN](https://github.com/TencentARC/GFPGAN) - Face enhancement
