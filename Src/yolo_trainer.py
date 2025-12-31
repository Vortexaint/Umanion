"""
Simple YOLO trainer/predictor wrapper used by the Uma project.

This module provides a light `YOLOTrainer` class that wraps the
`ultralytics.YOLO` API when available. If the `ultralytics` package is
not installed, the class provides safe no-op methods so imports don't fail.

The implementation is intentionally small: it supports `setup_classes`,
`train_model`, `load_model` and `predict` which are the methods used by
`src/main.py`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from ultralytics import YOLO  # type: ignore
    _HAS_ULTRALYTICS = True
except Exception:
    YOLO = None
    _HAS_ULTRALYTICS = False

try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False

import numpy as np


class YOLOTrainer:
    def __init__(self, project_dir: str = "yolo_project"):
        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.class_names: List[str] = []

    def setup_classes(self, classes: List[str]):
        """Store class names for future use (training / visualization)."""
        self.class_names = list(classes)
        classes_path = self.project_dir / "classes.txt"
        try:
            with open(classes_path, "w", encoding="utf-8") as f:
                for c in self.class_names:
                    f.write(f"{c}\n")
        except Exception:
            pass

    def train_model(self, epochs: int = 50, batch_size: int = 16, img_size: int = 640):
        """Start training using the Ultralitycs YOLO if available.

        This looks for `yolo_training/data.yaml` in the repository by default.
        If ultralytics isn't installed or the data file is missing, the method
        will print an explanatory message and return without error.
        """
        data_yaml = Path("yolo_training") / "data.yaml"
        if not _HAS_ULTRALYTICS:
            print("ultralytics not available; skipping YOLO training (install ultralytics)")
            return
        if not data_yaml.exists():
            print(f"Data config not found at {data_yaml}; create a data.yaml before training")
            return

        # Use a small default model if available (ultralytics will download if needed)
        try:
            model = YOLO("yolov8n.pt")
            model.train(data=str(data_yaml), epochs=epochs, imgsz=img_size, batch=batch_size,
                        project=str(self.project_dir), name="run")
            self.model = model
        except Exception as e:
            print(f"Error during training: {e}")

    def load_model(self, model_path: Optional[str] = None):
        """Load a YOLO model file (or default small model).

        If ultralytics is not installed this becomes a no-op.
        """
        path = model_path or "yolov8n.pt"

        # Prefer ultralytics YOLO (v8) if available
        if _HAS_ULTRALYTICS:
            try:
                self.model = YOLO(path)
                self._backend = "ultralytics"
                return
            except Exception as e:
                print(f"ultralytics YOLO failed to load '{path}': {e}")

        # Fallback: try torch.hub loading of YOLOv5 (custom weights)
        if _HAS_TORCH:
            try:
                # This will use the ultralytics/yolov5 hub repo and load custom weights
                hub_model = torch.hub.load('ultralytics/yolov5', 'custom', path, trust_repo=True)
                self.model = hub_model
                self._backend = "yolov5"
                return
            except Exception as e:
                print(f"torch.hub YOLOv5 failed to load '{path}': {e}")

        print("No supported YOLO backend available; install 'ultralytics' or 'torch' with access to torch.hub")

    def predict(self, image, conf: float = 0.25) -> List[dict]:
        """Run prediction on an image and return simple detection dicts.

        Returns a list of dicts: {"class": int, "label": str, "conf": float, "box": (x1,y1,x2,y2)}
        """
        if self.model is None:
            return []

        out: List[dict] = []
        try:
            if getattr(self, '_backend', None) == 'ultralytics':
                results = self.model.predict(source=image, conf=conf)
                for res in results:
                    boxes = getattr(res, "boxes", None)
                    if boxes is None:
                        continue
                    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else np.array([])
                    confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else np.array([])
                    cls_inds = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, "cls") else np.array([])

                    for i in range(len(xyxy)):
                        x1, y1, x2, y2 = map(int, xyxy[i][:4].tolist())
                        confv = float(confs[i]) if i < len(confs) else 0.0
                        cls_i = int(cls_inds[i]) if i < len(cls_inds) else -1
                        label = self.class_names[cls_i] if 0 <= cls_i < len(self.class_names) else str(cls_i)
                        out.append({
                            "class": cls_i,
                            "label": label,
                            "conf": confv,
                            "box": (x1, y1, x2, y2),
                        })

            elif getattr(self, '_backend', None) == 'yolov5':
                # torch.hub yolov5 model: call returns a Results object with .xyxy
                results = self.model(image)
                # results.xyxy is a list per image; take first
                try:
                    xyxy = results.xyxy[0].cpu().numpy()
                except Exception:
                    xyxy = np.array([])

                for row in xyxy:
                    x1, y1, x2, y2, confv, cls_i = row.tolist()
                    cls_i = int(cls_i)
                    label = self.class_names[cls_i] if 0 <= cls_i < len(self.class_names) else str(cls_i)
                    out.append({
                        "class": cls_i,
                        "label": label,
                        "conf": float(confv),
                        "box": (int(x1), int(y1), int(x2), int(y2)),
                    })
            else:
                # Unknown backend: try generic call
                results = self.model(image)
                try:
                    xyxy = results.xyxy[0].cpu().numpy()
                    for row in xyxy:
                        x1, y1, x2, y2, confv, cls_i = row.tolist()
                        cls_i = int(cls_i)
                        label = self.class_names[cls_i] if 0 <= cls_i < len(self.class_names) else str(cls_i)
                        out.append({
                            "class": cls_i,
                            "label": label,
                            "conf": float(confv),
                            "box": (int(x1), int(y1), int(x2), int(y2)),
                        })
                except Exception:
                    pass

            return out

        except Exception as e:
            print(f"Prediction error: {e}")
            return []


__all__ = ["YOLOTrainer"]
