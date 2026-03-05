"""Download required ONNX models for Hera."""
import os
import sys
import urllib.request

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


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    for filename, url in MODELS.items():
        dest = os.path.join(MODELS_DIR, filename)
        if os.path.exists(dest):
            print(f"{filename} already exists, skipping.")
            continue
        download_file(url, dest)

    print("\nAll models downloaded. buffalo_l will auto-download on first run.")


if __name__ == "__main__":
    main()
