import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Hera - AI Face Swap & Portrait Animation")
    parser.add_argument(
        "--reference", type=str, default=None,
        help="Path to reference face image"
    )
    parser.add_argument(
        "--no-enhance", action="store_true",
        help="Disable face enhancement for higher FPS"
    )
    parser.add_argument(
        "--detection-interval", type=int, default=3,
        help="Face detection frequency (every N frames)"
    )
    parser.add_argument(
        "--resolution", type=str, default="640x480",
        help="Capture resolution (WxH)"
    )
    parser.add_argument(
        "--generate", nargs=2, metavar=("SOURCE_IMAGE", "DRIVING_VIDEO"),
        help="Run headless generation: --generate <source_image> <driving_video>"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for headless generation (used with --generate)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Apply args to config
    from hera import config
    if args.reference:
        config.DEFAULT_REFERENCE_PATH = args.reference
    if args.no_enhance:
        config.ENHANCE_ENABLED = False
    config.FACE_DETECTION_INTERVAL = args.detection_interval

    w, h = args.resolution.split("x")
    config.CAPTURE_WIDTH = int(w)
    config.CAPTURE_HEIGHT = int(h)

    # Headless generation mode
    if args.generate:
        _run_headless_generate(args.generate[0], args.generate[1], args.output)
        return

    # Launch UI
    from hera.ui.app import HeraApp
    app = HeraApp()
    app.mainloop()


def _run_headless_generate(source_image, driving_video, output_path):
    import os
    from hera import config
    from hera.generate_pipeline import GeneratePipeline

    if output_path is not None:
        output_dir = os.path.dirname(output_path) or "."
    else:
        output_dir = config.GENERATE_OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)

    pipeline = GeneratePipeline()
    print("Loading LivePortrait models...")
    pipeline.load_models(progress_callback=lambda msg: print(f"  {msg}"))

    print(f"Source: {source_image}")
    print(f"Driving: {driving_video}")
    print(f"Output dir: {output_dir}")
    print("Generating...")

    done_event = __import__("threading").Event()

    def _progress(status, detail):
        if status == "generating":
            sys.stdout.write("\r  Generating...")
            sys.stdout.flush()
        elif status == "done":
            sys.stdout.write("\r  Done.          \n")
            sys.stdout.flush()

    def _done():
        done_event.set()

    pipeline.generate(
        source_image, driving_video,
        output_dir=output_dir,
        progress_callback=_progress,
        done_callback=_done,
    )

    done_event.wait()
    print()

    if pipeline.error:
        print(f"Error: {pipeline.error}")
        sys.exit(1)

    if pipeline.output_path and os.path.exists(pipeline.output_path):
        print(f"Done! Output saved to: {pipeline.output_path}")
    else:
        print("Generation failed - no output produced.")
        sys.exit(1)


if __name__ == "__main__":
    main()
