"""Capture primary monitor and run OCR using `OCR` from `Ocr.py`.

Saves a screenshot to `Debug/monitor1.png` and OCR result to `Debug/ocr_output.txt` by default.
"""
import os
import argparse
import numpy as np
from PIL import Image
import ctypes
from ctypes import wintypes

try:
    import mss
except Exception:
    mss = None

from Ocr import OCR


def _get_primary_monitor_rect_windows():
    """Return primary monitor rect as (left, top, right, bottom) on Windows, or None."""
    user32 = ctypes.windll.user32
    # Define RECT and MONITORINFOEXW
    class RECT(ctypes.Structure):
        _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG), ("right", wintypes.LONG), ("bottom", wintypes.LONG)]

    class MONITORINFOEXW(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", RECT), ("rcWork", RECT), ("dwFlags", wintypes.DWORD), ("szDevice", wintypes.WCHAR * 32)]

    MonitorEnumProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC, ctypes.POINTER(RECT), wintypes.LPARAM)

    primary_rect = {}

    def _callback(hMonitor, hdcMonitor, lprcMonitor, dwData):
        mi = MONITORINFOEXW()
        mi.cbSize = ctypes.sizeof(mi)
        res = ctypes.windll.user32.GetMonitorInfoW(hMonitor, ctypes.byref(mi))
        if not res:
            return True
        MONITORINFOF_PRIMARY = 1
        if mi.dwFlags & MONITORINFOF_PRIMARY:
            primary_rect['left'] = mi.rcMonitor.left
            primary_rect['top'] = mi.rcMonitor.top
            primary_rect['right'] = mi.rcMonitor.right
            primary_rect['bottom'] = mi.rcMonitor.bottom
            # stop enumeration
            return False
        return True

    enum_proc = MonitorEnumProc(_callback)
    user32.EnumDisplayMonitors(0, 0, enum_proc, 0)

    if primary_rect:
        return (primary_rect['left'], primary_rect['top'], primary_rect['right'], primary_rect['bottom'])
    return None


def capture_primary_monitor() -> Image.Image:
    if mss is None:
        raise RuntimeError("Missing dependency 'mss'. Install with: pip install mss")

    with mss.mss() as sct:
        monitors = sct.monitors

        # Try to detect primary monitor on Windows using WinAPI
        monitor_choice = None
        if os.name == 'nt':
            try:
                rect = _get_primary_monitor_rect_windows()
                if rect:
                    left, top, right, bottom = rect
                    width = right - left
                    height = bottom - top
                    # Find matching monitor in mss.monitors
                    for m in monitors[1:]:
                        if m.get('left') == left and m.get('top') == top and m.get('width') == width and m.get('height') == height:
                            monitor_choice = m
                            break
            except Exception:
                monitor_choice = None

        # Fallbacks
        if monitor_choice is None:
            # prefer monitors[1] (common primary), else virtual monitors[0]
            monitor_choice = monitors[1] if len(monitors) >= 2 else monitors[0]

        sct_img = sct.grab(monitor_choice)
        img = Image.frombytes("RGB", sct_img.size, sct_img.rgb)
        return img


def main():
    parser = argparse.ArgumentParser(description="Capture primary monitor and run OCR")
    parser.add_argument("--out", "-o", default="Debug", help="Output directory (default: Debug)")
    parser.add_argument("--image-name", default="monitor1.png", help="Screenshot filename")
    parser.add_argument("--text-name", default="ocr_output.txt", help="OCR text filename")
    parser.add_argument("--rect", help="Crop rectangle. Examples: 'x=261 y=82 to x=369 y=136' or 'left,top,right,bottom' or four numbers")
    parser.add_argument("--find-text", dest="find_text", help="Text to search for in the OCR output")
    args = parser.parse_args()

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    img = capture_primary_monitor()

    img_path = os.path.join(out_dir, args.image_name)
    img.save(img_path)

    # If a rect is provided, try to parse numbers and crop the image for OCR
    cropped_img = img
    # default crop offset (used to translate OCR boxes to screen coordinates)
    crop_offset = (0, 0)
    if getattr(args, 'rect', None):
        import re

        nums = re.findall(r"-?\d+", args.rect)
        if len(nums) >= 4:
            # take first 4 numbers as left, top, right, bottom
            left, top, right, bottom = map(int, nums[:4])
            try:
                cropped_img = img.crop((left, top, right, bottom))
                # record crop offset so coordinates map to screen coordinates
                crop_offset = (left, top)
                # save cropped image next to full screenshot
                base, ext = os.path.splitext(args.image_name)
                crop_name = f"{base}_crop{ext}"
                crop_path = os.path.join(out_dir, crop_name)
                cropped_img.save(crop_path)
                print(f"Saved cropped screenshot to: {crop_path}")
            except Exception as e:
                print(f"Failed to crop image with rect '{args.rect}': {e}")
        else:
            print(f"Unable to parse rect '{args.rect}'; need four integers.")

    # Convert PIL image (cropped or full) to numpy array for OCR class
    np_img = np.array(cropped_img)

    ocr = OCR()
    text = ocr.extract_text(np_img)

    text_path = os.path.join(out_dir, args.text_name)
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Saved screenshot to: {img_path}")
    print(f"Saved OCR text to: {text_path}")
    # If requested, search for specific text in OCR output and report coordinates
    if getattr(args, 'find_text', None):
        query = args.find_text
        found_any = False
        q_lower = query.lower()
        # Try to use word-level boxes if OCR supports it
        try:
            words = ocr.extract_words_with_boxes(np_img)
            matches = []
            for w in words:
                if q_lower in w['text'].lower():
                    # boxes are relative to the image passed to OCR (cropped image if used)
                    left = w['left'] + crop_offset[0]
                    top = w['top'] + crop_offset[1]
                    right = left + w['width']
                    bottom = top + w['height']
                    matches.append((w['text'], left, top, right, bottom, w.get('conf', -1)))

            if matches:
                for i, m in enumerate(matches, start=1):
                    print(f"Found '{m[0]}' at left={m[1]} top={m[2]} right={m[3]} bottom={m[4]} (conf={m[5]})")
                    # Attempt to save a cropped image of the found box from the full screenshot
                    try:
                        # crop from the full screenshot so coordinates are absolute
                        crop_box = (m[1], m[2], m[3], m[4])
                        cropped = img.crop(crop_box)
                        # create safe filename fragment from the text
                        import re

                        safe_text = re.sub(r"[^A-Za-z0-9_-]", "_", m[0])[:30]
                        base, ext = os.path.splitext(args.image_name)
                        found_name = f"{base}_found_{i}_{safe_text}{ext}"
                        found_path = os.path.join(out_dir, found_name)
                        cropped.save(found_path)
                        print(f"Saved crop for match {i} to: {found_path}")
                    except Exception as e:
                        print(f"Failed to save crop for match {i}: {e}")
                found_any = True
            else:
                # fallback to plain text search
                if query in text or q_lower in text.lower():
                    print(f"Found '{query}' in OCR output (no box-level match)")
                    found_any = True
        except Exception:
            # If box extraction isn't available or fails, fallback to plain text search
            if query in text or q_lower in text.lower():
                print(f"Found '{query}' in OCR output (box lookup failed)")
                found_any = True

        if not found_any:
            print(f"Did not find '{query}' in OCR output.")


if __name__ == "__main__":
    main()
