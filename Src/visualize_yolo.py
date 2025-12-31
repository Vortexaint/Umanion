import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def load_class_names(path):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def yolo_labels_to_boxes(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r", encoding="utf-8") as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) < 5:
                continue
            cls = parts[0]
            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            x1 = int((xc - w / 2.0) * img_w)
            y1 = int((yc - h / 2.0) * img_h)
            x2 = int((xc + w / 2.0) * img_w)
            y2 = int((yc + h / 2.0) * img_h)
            boxes.append((int(cls), x1, y1, x2, y2))
    return boxes


def draw_boxes(img, boxes, class_names=None, color=(0, 255, 0), thickness=2):
    out = img.copy()
    for cls, x1, y1, x2, y2 in boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = str(cls) if class_names is None or cls >= len(class_names) else class_names[cls]
        # background for text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def find_label_for_image(image_path, labels_dir=None):
    p = Path(image_path)
    # default label name: same stem + .txt
    label_name = p.with_suffix(".txt").name
    if labels_dir:
        candidate = Path(labels_dir) / label_name
        return str(candidate)
    # try same folder as image
    candidate = p.with_suffix(".txt")
    return str(candidate)


def main():
    ap = argparse.ArgumentParser(description="Visualize YOLO-format boxes on an image")
    ap.add_argument("--image", required=True, help="Path to image file")
    ap.add_argument("--labels", help="Path to corresponding label .txt file (optional). If omitted, script looks beside image")
    ap.add_argument("--labels-dir", help="Directory where label files live (optional)")
    ap.add_argument("--class-names", help="Optional file listing class names (one per line)")
    ap.add_argument("--output", help="Save visualized image to this path (optional)")
    ap.add_argument("--no-show", action="store_true", help="Do not open a window; just save")
    args = ap.parse_args()

    img_path = args.image
    if not os.path.exists(img_path):
        raise SystemExit(f"Image not found: {img_path}")

    img = cv2.imread(img_path)
    if img is None:
        raise SystemExit(f"Failed to read image: {img_path}")
    h, w = img.shape[:2]

    if args.labels:
        label_path = args.labels
    else:
        label_path = find_label_for_image(img_path, args.labels_dir)

    class_names = load_class_names(args.class_names) if args.class_names else None

    boxes = yolo_labels_to_boxes(label_path, w, h)

    out = draw_boxes(img, boxes, class_names=class_names)

    if args.output:
        cv2.imwrite(args.output, out)
        print(f"Saved visualization to {args.output}")
    if not args.no_show:
        win = "YOLO Visual"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.imshow(win, out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
