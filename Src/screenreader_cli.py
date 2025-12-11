"""Command-line interface for screen capture + OCR.

This module contains the `main()` function extracted from the original
`ScreenReader.py`. It uses the `capture` and `live_clone` modules.
"""
import os
import argparse
import re
import time
import numpy as np
from Ocr import OCR
from capture import capture_primary_monitor
from live_clone import LiveCloneController


def main(argv=None):
    parser = argparse.ArgumentParser(description="Capture primary monitor and run OCR")
    parser.add_argument("--out", "-o", default="Debug", help="Output directory (default: Debug)")
    parser.add_argument("--image-name", default="monitor1.png", help="Screenshot filename")
    parser.add_argument("--text-name", default="ocr_output.txt", help="OCR text filename")
    parser.add_argument(
        "--rect",
        help="Crop rectangle. Examples: 'x=261 y=82 to x=369 y=136' or 'left,top,right,bottom' or four numbers",
    )
    parser.add_argument("--find-text", dest="find_text", help="Text to search for in the OCR output")
    parser.add_argument("--live", action="store_true", help="Run live clone window with continuous OCR and overlay")
    args = parser.parse_args(argv)

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    img = capture_primary_monitor()

    img_path = os.path.join(out_dir, args.image_name)
    img.save(img_path)

    cropped_img = img
    crop_offset = (0, 0)
    if getattr(args, 'rect', None):
        nums = re.findall(r"-?\d+", args.rect)
        if len(nums) >= 4:
            left, top, right, bottom = map(int, nums[:4])
            try:
                cropped_img = img.crop((left, top, right, bottom))
                crop_offset = (left, top)
                base, ext = os.path.splitext(args.image_name)
                crop_name = f"{base}_crop{ext}"
                crop_path = os.path.join(out_dir, crop_name)
                cropped_img.save(crop_path)
                print(f"Saved cropped screenshot to: {crop_path}")
            except Exception as e:
                print(f"Failed to crop image with rect '{args.rect}': {e}")
        else:
            print(f"Unable to parse rect '{args.rect}'; need four integers.")

    np_img = np.array(cropped_img)

    ocr = OCR()
    text = ocr.extract_text(np_img)

    text_path = os.path.join(out_dir, args.text_name)
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Saved screenshot to: {img_path}")
    print(f"Saved OCR text to: {text_path}")

    if getattr(args, 'live', False):
        ctrl = LiveCloneController(find_text=args.find_text, scale=0.5, interval=700)
        ctrl.start()
        try:
            while ctrl._thread and ctrl._thread.is_alive():
                time.sleep(0.2)
        except KeyboardInterrupt:
            ctrl.stop()
        return

    if getattr(args, 'find_text', None):
        query = args.find_text
        found_any = False
        q_lower = query.lower()
        try:
            words = ocr.extract_words_with_boxes(np_img)
            matches = []
            for w in words:
                if q_lower in w['text'].lower():
                    left = w['left'] + crop_offset[0]
                    top = w['top'] + crop_offset[1]
                    right = left + w['width']
                    bottom = top + w['height']
                    matches.append((w['text'], left, top, right, bottom, w.get('conf', -1)))

            if matches:
                m = matches[0]
                print(f"Found '{m[0]}' at left={m[1]} top={m[2]} right={m[3]} bottom={m[4]} (conf={m[5]})")
                try:
                    crop_box = (m[1], m[2], m[3], m[4])
                    cropped = img.crop(crop_box)
                    safe_text = re.sub(r"[^A-Za-z0-9_-]", "_", m[0])[:30]
                    base, ext = os.path.splitext(args.image_name)
                    found_name = f"{base}_found{ext}"
                    found_path = os.path.join(out_dir, found_name)
                    cropped.save(found_path)
                    print(f"Saved crop to: {found_path} (overwritten)")
                except Exception as e:
                    print(f"Failed to save crop: {e}")
                found_any = True
            else:
                if query in text or q_lower in text.lower():
                    print(f"Found '{query}' in OCR output (no box-level match)")
                    found_any = True
        except Exception:
            if query in text or q_lower in text.lower():
                print(f"Found '{query}' in OCR output (box lookup failed)")
                found_any = True

        if not found_any:
            print(f"Did not find '{query}' in OCR output.")


if __name__ == "__main__":
    main()
