"""Capture sample screenshots to populate YOLO dataset folders.

Usage:
  python src/capture_yolo_samples.py --project=yolo_project --train=5 --val=1

This will save screenshots into the project's data/images/train and data/images/val
and create corresponding empty label files so ultralytics can find files.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from screen_capture import ScreenCapture


def main(project_dir: str = "yolo_project", train_count: int = 5, val_count: int = 1):
    sc = ScreenCapture()
    base = Path(project_dir) / "data"
    img_train = base / "images" / "train"
    img_val = base / "images" / "val"
    lbl_train = base / "labels" / "train"
    lbl_val = base / "labels" / "val"

    for p in [img_train, img_val, lbl_train, lbl_val]:
        p.mkdir(parents=True, exist_ok=True)

    idx = 0
    print(f"Capturing {train_count} train images and {val_count} val images into {base}")

    for i in range(train_count):
        img = sc.capture_monitor(sc.get_primary_monitor())
        arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        fname = img_train / f"sample_train_{i:03d}.jpg"
        cv2.imwrite(str(fname), arr)
        # create empty label file
        (lbl_train / f"sample_train_{i:03d}.txt").write_text("")
        print(f"Saved {fname}")

    for i in range(val_count):
        img = sc.capture_monitor(sc.get_primary_monitor())
        arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        fname = img_val / f"sample_val_{i:03d}.jpg"
        cv2.imwrite(str(fname), arr)
        (lbl_val / f"sample_val_{i:03d}.txt").write_text("")
        print(f"Saved {fname}")

    print("Done. You can now run training (note: empty labels mean no objects; label images before serious training).")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='yolo_project')
    parser.add_argument('--train', type=int, default=5)
    parser.add_argument('--val', type=int, default=1)
    args = parser.parse_args()
    main(project_dir=args.project, train_count=args.train, val_count=args.val)
