import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Hera - Real-Time AI Face Swap")
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

    # Launch UI
    from hera.ui.app import HeraApp
    app = HeraApp()
    app.mainloop()


if __name__ == "__main__":
    main()
