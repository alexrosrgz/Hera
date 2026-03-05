"""Download required ONNX models for Hera."""
import os
import ssl
import sys
import urllib.request

# Fix SSL certificate issues on macOS
ssl._create_default_https_context = ssl._create_unverified_context

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

MODELS = {
    "inswapper_128_fp16.onnx": "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx",
    "GFPGANv1.4.onnx": "https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.onnx",
}


def download_file(url, dest):
    print(f"Downloading {os.path.basename(dest)}...")

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  {pct:.1f}% ({mb:.1f}/{total_mb:.1f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print("\n  Done.")


LIVEPORTRAIT_PRETRAINED_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "vendor", "LivePortrait", "pretrained_weights"
)

LIVEPORTRAIT_REPO = "KwaiVGI/LivePortrait"


def download_liveportrait_models():
    """Download LivePortrait model weights from HuggingFace via snapshot_download.

    The HF repo KwaiVGI/LivePortrait contains top-level dirs: liveportrait/, insightface/, etc.
    We download into pretrained_weights/ so the final layout is:
        pretrained_weights/liveportrait/base_models/*.pth
        pretrained_weights/insightface/models/buffalo_l/*.onnx
    This matches what LivePortrait's default config paths expect.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub not installed. Skipping LivePortrait models.")
        print("Install with: pip install huggingface_hub")
        return

    expected = os.path.join(LIVEPORTRAIT_PRETRAINED_DIR, "liveportrait", "base_models", "appearance_feature_extractor.pth")
    if os.path.exists(expected):
        print("LivePortrait models already downloaded, skipping.")
        return

    print("Downloading LivePortrait models (this may take a while)...")
    try:
        snapshot_download(
            repo_id=LIVEPORTRAIT_REPO,
            local_dir=LIVEPORTRAIT_PRETRAINED_DIR,
        )
        print("  LivePortrait models downloaded.")
    except Exception as e:
        print(f"  Failed to download LivePortrait models: {e}")
        print(f"  You may need to download manually from https://huggingface.co/{LIVEPORTRAIT_REPO}")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Download ONNX models (face swap + enhancement)
    for filename, url in MODELS.items():
        dest = os.path.join(MODELS_DIR, filename)
        if os.path.exists(dest):
            print(f"{filename} already exists, skipping.")
            continue
        download_file(url, dest)

    print("\nONNX models done. buffalo_l will auto-download on first run.")

    # Download LivePortrait models
    print("\n--- LivePortrait Models ---")
    download_liveportrait_models()

    print("\nAll models downloaded.")


if __name__ == "__main__":
    main()
