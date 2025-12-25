from pathlib import Path
from PIL import Image
from screen_capture import ScreenCapture
from ocr import extract_number
import argparse


DEFAULT_LEFT = 307
DEFAULT_TOP = 722
DEFAULT_RIGHT = 745
DEFAULT_BOTTOM = 745


def capture_and_read(left: int, top: int, right: int, bottom: int, monitor: int = 1):
    width = right - left
    height = bottom - top
    sc = ScreenCapture()
    img = sc.capture_region(left, top, width, height, monitor_num=monitor)

    Path("debug").mkdir(exist_ok=True)

    # Split into 5 equal vertical segments
    stats = ["speed", "stamina", "power", "guts", "wit"]
    results = {}
    seg_w = max(1, width // 5)

    for i, name in enumerate(stats):
        seg_left = i * seg_w
        # last segment takes remaining width
        seg_right = (i + 1) * seg_w if i < 4 else width
        crop = img.crop((seg_left, 0, seg_right, height))
        # save debug image
        out_path = Path("debug") / f"stat_{i+1}_{name}.png"
        crop.save(out_path)

        val = extract_number(crop)
        results[name] = val

    return results


def main():
    parser = argparse.ArgumentParser(description="Capture stats from a screen region and OCR five stat values")
    parser.add_argument("--left", type=int, default=DEFAULT_LEFT)
    parser.add_argument("--top", type=int, default=DEFAULT_TOP)
    parser.add_argument("--right", type=int, default=DEFAULT_RIGHT)
    parser.add_argument("--bottom", type=int, default=DEFAULT_BOTTOM)
    parser.add_argument("--monitor", type=int, default=1)
    args = parser.parse_args()

    res = capture_and_read(args.left, args.top, args.right, args.bottom, monitor=args.monitor)
    # Print in order
    order = ["speed", "stamina", "power", "guts", "wit"]
    print("Detected stats:")
    for k in order:
        print(f"{k}: {res.get(k, -1)}")


if __name__ == "__main__":
    main()
